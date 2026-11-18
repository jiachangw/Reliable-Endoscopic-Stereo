# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
from .lightstereo import LightStereo
import time
import torch
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard, write_wandb
from stereo.datasets import build_dataloader

__all__ = {
    'LightStereo': LightStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss
        norm = False
        for cur_cfg in self.cfgs.DATA_CONFIG.DATA_TRANSFORM.TRAINING:
            if hasattr(cur_cfg, 'NAME') and cur_cfg.NAME == 'NormalizeImage':
                norm = True


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
                data[k] = v.to(self.local_rank,non_blocking=True) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
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
                if norm:
                    tb_info['image/train/image'] = torch.cat([self.denormalize_image(data['left'][0]), self.denormalize_image(data['right'][0])], dim=1)
                else:
                    tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr,
                            'scalar/train/total_loss': total_loss / (i + 1)})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_wandb(self.tb_writer, tb_info, total_iter)

    def build_train_loader(self):
        train_set, train_loader, train_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='training',
            prefetch_factor=2)
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        eval_set, eval_loader, eval_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating',
            prefetch_factor=2)
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler


    # def train_one_epoch(self, current_epoch, tbar):
    #     start_epoch = self.last_epoch + 1
    #     logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
    #     total_loss = 0.0
    #     # 修正：使用更健壮的loss_func获取方式
    #     loss_func = self.model.module.get_loss if (self.args.dist_mode and hasattr(self.model, 'module')) else self.model.get_loss
    #     norm = False
    #     for cur_cfg in self.cfgs.DATA_CONFIG.DATA_TRANSFORM.TRAINING:
    #         if hasattr(cur_cfg, 'NAME') and cur_cfg.NAME == 'NormalizeImage':
    #             norm = True

    #     # 时间统计变量
    #     total_data_time = 0.0
    #     total_infer_time = 0.0
    #     total_loss_time = 0.0
    #     total_backward_time = 0.0
    #     total_grad_clip_time = 0.0
    #     total_opt_step_time = 0.0
    #     total_scheduler_time = 0.0
    #     total_grad_zero_time = 0.0

    #     # 异常检测
    #     slow_data_iterations = 0
    #     slow_opt_iterations = 0

    #     # 修正1：使用enumerate遍历数据加载器，避免手动迭代器的内存泄漏
    #     for i, data in enumerate(self.train_loader):
    #         total_iter = current_epoch * len(self.train_loader) + i
    #         if total_iter >= self.max_iter:
    #             break

    #         iter_start_time = time.time()

    #         # 修正2：将梯度零初始化移到**前向传播前**（常规流程），且放在数据加载后
    #         grad_zero_start = time.time()
    #         self.optimizer.zero_grad(set_to_none=True)  # 关键优化：使用set_to_none=True，减少显存占用
    #         lr = self.optimizer.param_groups[0]['lr']
    #         grad_zero_time = time.time() - grad_zero_start
    #         total_grad_zero_time += grad_zero_time

    #         # 1. 数据加载（已整合到enumerate中，保留时间统计）
    #         data_load_start = time.time()
    #         for k, v in data.items():
    #             if torch.is_tensor(v):
    #                 # 修正3：分布式训练时，确保设备正确且非阻塞
    #                 data[k] = v.to(self.local_rank, non_blocking=True)
    #         data_time = time.time() - data_load_start
    #         total_data_time += data_time

    #         # 检测数据加载异常
    #         if data_time > 0.1:
    #             slow_data_iterations += 1
    #             if data_time > 1.0:
    #                 self.logger.warning(f"Iter {i}: 数据加载异常耗时 {data_time*1000:.1f}ms")

    #         # 2. 模型推理
    #         infer_start = time.time()
    #         with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
    #             model_pred = self.model(data)
    #         infer_time = time.time() - infer_start
    #         total_infer_time += infer_time

    #         # 3. 损失计算
    #         loss_start = time.time()
    #         with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
    #             loss, tb_info = loss_func(model_pred, data)
    #         loss_time = time.time() - loss_start
    #         total_loss_time += loss_time

    #         # 4. 反向传播
    #         backward_start = time.time()
    #         self.scaler.scale(loss).backward()
    #         self.scaler.unscale_(self.optimizer)
    #         backward_time = time.time() - backward_start
    #         total_backward_time += backward_time

    #         # 5. 梯度裁剪
    #         grad_clip_start = time.time()
    #         if self.clip_gard is not None:
    #             self.clip_gard(self.model)
    #         grad_clip_time = time.time() - grad_clip_start
    #         total_grad_clip_time += grad_clip_time

    #         if grad_clip_time > 0.05:
    #             slow_opt_iterations += 1

    #         # 6. 优化器步骤
    #         opt_step_start = time.time()
    #         # 修正4：添加AMP缩放器的异常处理，避免状态膨胀
    #         try:
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         except Exception as e:
    #             self.logger.error(f"Opt step error: {e}")
    #             # 重置scaler，防止状态异常
    #             self.scaler = torch.cuda.amp.GradScaler()
    #         opt_step_time = time.time() - opt_step_start
    #         total_opt_step_time += opt_step_time

    #         # 7. 调度器步骤
    #         scheduler_start = time.time()
    #         with self.warmup_scheduler.dampening():
    #             if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
    #                 self.scheduler.step()
    #         scheduler_time = time.time() - scheduler_start
    #         total_scheduler_time += scheduler_time

    #         # 修正5：提高CUDA缓存清理频率，且只在主进程执行
    #         if i % 100 == 0 and self.local_rank == 0:
    #             torch.cuda.empty_cache()
    #             # 额外优化：清理Python的垃圾回收
    #             import gc
    #             gc.collect()

    #         # 迭代总时间
    #         iter_total_time = time.time() - iter_start_time

    #         total_loss += loss.item()
    #         trained_time_past_all = tbar.format_dict['elapsed']
    #         # 修正6：修复剩余时间计算的除数为零问题
    #         denominator = (total_iter + 1 - start_epoch * len(self.train_loader))
    #         denominator = denominator if denominator != 0 else 1
    #         single_iter_second = trained_time_past_all / denominator
    #         remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)

    #         if total_iter % logger_iter_interval == 0:
    #             # 计算平均时间
    #             avg_data_time = total_data_time / (i + 1)
    #             avg_infer_time = total_infer_time / (i + 1)
    #             avg_loss_time = total_loss_time / (i + 1)
    #             avg_backward_time = total_backward_time / (i + 1)
    #             avg_grad_clip_time = total_grad_clip_time / (i + 1)
    #             avg_opt_step_time = total_opt_step_time / (i + 1)
    #             avg_scheduler_time = total_scheduler_time / (i + 1)
    #             avg_grad_zero_time = total_grad_zero_time / (i + 1)

    #             # 当前迭代的时间
    #             cur_data_time = data_time
    #             cur_infer_time = infer_time
    #             cur_loss_time = loss_time
    #             cur_backward_time = backward_time
    #             cur_grad_clip_time = grad_clip_time
    #             cur_opt_step_time = opt_step_time
    #             cur_scheduler_time = scheduler_time
    #             cur_grad_zero_time = grad_zero_time

    #             # 计算平均迭代总时间
    #             avg_iter_time = (avg_data_time + avg_infer_time + avg_loss_time +
    #                             avg_backward_time + avg_grad_clip_time + avg_opt_step_time +
    #                             avg_scheduler_time + avg_grad_zero_time)

    #             # 计算当前迭代的总时间
    #             cur_iter_time = (cur_data_time + cur_infer_time + cur_loss_time +
    #                             cur_backward_time + cur_grad_clip_time + cur_opt_step_time +
    #                             cur_scheduler_time + cur_grad_zero_time)

    #             # 计算时间占比（添加零值判断，避免除零错误）
    #             data_pct = (cur_data_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             infer_pct = (cur_infer_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             loss_pct = (cur_loss_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             backward_pct = (cur_backward_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             grad_clip_pct = (cur_grad_clip_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             opt_step_pct = (cur_opt_step_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             scheduler_pct = (cur_scheduler_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0
    #             grad_zero_pct = (cur_grad_zero_time / cur_iter_time) * 100 if cur_iter_time > 0 else 0

    #             message = (
    #                 'Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
    #                 'Loss:{:#.6g}({:#.6g}) LR:{:.4e}\n'
    #                 'Current Time (ms) [Avg Time]:\n'
    #                 '  Data: {:>5.1f} [{:>5.1f}] ({:>4.1f}%) | '
    #                 'Infer: {:>5.1f} [{:>5.1f}] ({:>4.1f}%) | '
    #                 'Loss: {:>5.1f} [{:>5.1f}] ({:>4.1f}%)\n'
    #                 '  Backward: {:>5.1f} [{:>5.1f}] ({:>4.1f}%) | '
    #                 'GradClip: {:>5.1f} [{:>5.1f}] ({:>4.1f}%) | '
    #                 'OptStep: {:>5.1f} [{:>5.1f}] ({:>4.1f}%)\n'
    #                 '  Scheduler: {:>5.1f} [{:>5.1f}] ({:>4.1f}%) | '
    #                 'GradZero: {:>5.1f} [{:>5.1f}] ({:>4.1f}%)\n'
    #                 '  Iter Total: {:>7.2f} [Avg: {:>7.2f}] ms | '
    #                 'SlowData/Opt: {}/{}\n'
    #                 '  Remaining: {} | '
    #                 'Speed: {:>4.1f} it/s [Avg: {:>4.1f} it/s]'
    #             ).format(
    #                 current_epoch, self.total_epochs, i, len(self.train_loader),
    #                 loss.item(), total_loss / (i + 1), lr,
    #                 cur_data_time * 1000, avg_data_time * 1000, data_pct,
    #                 cur_infer_time * 1000, avg_infer_time * 1000, infer_pct,
    #                 cur_loss_time * 1000, avg_loss_time * 1000, loss_pct,
    #                 cur_backward_time * 1000, avg_backward_time * 1000, backward_pct,
    #                 cur_grad_clip_time * 1000, avg_grad_clip_time * 1000, grad_clip_pct,
    #                 cur_opt_step_time * 1000, avg_opt_step_time * 1000, opt_step_pct,
    #                 cur_scheduler_time * 1000, avg_scheduler_time * 1000, scheduler_pct,
    #                 cur_grad_zero_time * 1000, avg_grad_zero_time * 1000, grad_zero_pct,
    #                 cur_iter_time * 1000, avg_iter_time * 1000,
    #                 slow_data_iterations, slow_opt_iterations,
    #                 tbar.format_interval(remaining_second_all),
    #                 1.0 / cur_iter_time if cur_iter_time > 0 else 0,
    #                 1.0 / avg_iter_time if avg_iter_time > 0 else 0
    #             )
    #             self.logger.info(message)

    #         # 可视化相关代码
    #         if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
    #             if norm:
    #                 tb_info['image/train/image'] = torch.cat([self.denormalize_image(data['left'][0]), self.denormalize_image(data['right'][0])], dim=1)
    #             else:
    #                 tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
    #             tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

    #         tb_info.update({'scalar/train/lr': lr,
    #                         'scalar/train/total_loss': total_loss / (i + 1)})
    #         if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
    #             write_wandb(self.tb_writer, tb_info, total_iter)

    #     # 每个epoch结束时输出总结（添加i的零值判断）
    #     if i > 0:
    #         total_iter_time = (total_data_time + total_infer_time + total_loss_time +
    #                         total_backward_time + total_grad_clip_time + total_opt_step_time +
    #                         total_scheduler_time + total_grad_zero_time)
    #         # 避免除零错误
    #         total_iter_time = total_iter_time if total_iter_time > 0 else 1

    #         self.logger.info('=' * 100)
    #         self.logger.info(f'Epoch {current_epoch} Time Summary:')
    #         self.logger.info(f'  Data Time:      {total_data_time:.3f}s ({total_data_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  Infer Time:     {total_infer_time:.3f}s ({total_infer_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  Loss Time:      {total_loss_time:.3f}s ({total_loss_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  Backward Time:  {total_backward_time:.3f}s ({total_backward_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  GradClip Time:  {total_grad_clip_time:.3f}s ({total_grad_clip_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  OptStep Time:   {total_opt_step_time:.3f}s ({total_opt_step_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  Scheduler Time: {total_scheduler_time:.3f}s ({total_scheduler_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  GradZero Time:  {total_grad_zero_time:.3f}s ({total_grad_zero_time/total_iter_time*100:.1f}%)')
    #         self.logger.info(f'  Total Time:     {total_iter_time:.3f}s')
    #         self.logger.info(f'  Avg Iter Time:  {total_iter_time/(i+1):.3f}s')
    #         self.logger.info(f'  Speed:          {(i+1)/total_iter_time:.2f} it/s')
    #         self.logger.info(f'  Slow Data Iters: {slow_data_iterations}/{i+1} ({slow_data_iterations/(i+1)*100:.1f}%)')
    #         self.logger.info(f'  Slow Opt Iters:  {slow_opt_iterations}/{i+1} ({slow_opt_iterations/(i+1)*100:.1f}%)')
    #         self.logger.info('=' * 100)
