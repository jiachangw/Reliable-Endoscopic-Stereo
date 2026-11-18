from stereo.modeling.trainer_template import TrainerTemplate
from .core.monster import Monster

import time
import glob
import torch
import torch.nn as nn
import torch.distributed as dist
import os

from functools import partial
from stereo.datasets import build_dataloader
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_wandb
from stereo.utils.warmup import LinearWarmup
from stereo.utils.clip_grad import ClipGrad
from stereo.utils.lamb import Lamb
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric,loss_metric

__all__ = {
    'Monster': Monster,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def train_one_epoch(self, current_epoch, tbar):
        # torch.autograd.set_detect_anomaly(True)  # 会降低性能，仅调试时使用
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP,dtype=torch.bfloat16):
                model_pred = self.model(data,iters=self.cfgs.MODEL.train_iters)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # 梯度剪裁
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
            # torch.cuda.empty_cache()

            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()
            total_iter = current_epoch * len(self.train_loader) + i
            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr,
                            'scalar/train/total_loss': total_loss / (i + 1)})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_wandb(self.tb_writer, tb_info, total_iter)


    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):

        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
            'loss' : loss_metric
        }

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank

        epoch_metrics = {}
        for k in evaluator_cfgs.METRIC:
            epoch_metrics[k] = {'indexes': [], 'values': []}

        for i, data in enumerate(self.eval_loader):
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP,dtype=torch.bfloat16):
                infer_start = time.time()
                model_pred = self.model(data,iters=self.cfgs.MODEL.valid_iters)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            for m in evaluator_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                if m == 'loss':
                    res = loss_metric(model_pred,data,self.cfgs.MODEL.SMTH_LOSS_WEIGHT)
                else:
                    metric_func = metric_func_dict[m]
                    res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                    }
                    write_wandb(self.tb_writer, tb_info, current_epoch * len(self.eval_loader) + i)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())

        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v.item()

            write_wandb(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")
        return results

    def build_model(self, model):
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.local_rank)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

        # load pretrained model
        if self.cfgs.MODEL.PRETRAINED_MODEL:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError

            common_utils.load_params_from_file(
                model, self.cfgs.MODEL.PRETRAINED_MODEL,
                device='cuda:%d' % self.local_rank,
                dist_mode=self.args.dist_mode, logger=self.logger,
                strict=self.cfgs.MODEL.get('STRICT_LOAD', True))



            # common_utils.load_params_from_file(
            #     model, self.cfgs.MODEL.PRETRAINED_MODEL, device='cuda:%d' % self.local_rank,
            #     dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        return model
