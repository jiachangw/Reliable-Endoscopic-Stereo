# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
from .igev_stereo import IGEVStereo

import torch
import torch.nn as nn
import os

from stereo.utils import common_utils


__all__ = {
    'IGEV': IGEVStereo,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    # def build_model(self, model):
    #     if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
    #         model = common_utils.freeze_bn(model)
    #         self.logger.info('Freeze the batch normalization layers')

    #     if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #         self.logger.info('Convert batch norm to sync batch norm')
    #     model = model.to(self.local_rank)

    #     if self.args.dist_mode:
    #         model = nn.parallel.DistributedDataParallel(
    #             model, device_ids=[self.local_rank], output_device=self.local_rank,
    #             find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

    #     # load pretrained model
    #     if self.cfgs.MODEL.PRETRAINED_MODEL:
    #         self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
    #         if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
    #             raise FileNotFoundError

    #         checkpoint = torch.load(self.cfgs.MODEL.PRETRAINED_MODEL, map_location='cuda:%d' % self.local_rank)

    #         if 'model_state' in checkpoint.keys():
    #             checkpoint = checkpoint['model_state']


    #         ckpt = {}
    #         if self.args.dist_mode:
    #             model.load_state_dict(checkpoint,strict=False)
    #         else:
    #             for k,v in checkpoint.items():
    #                 name = k[7:] if k.startswith('module.') else k
    #                 ckpt[name] = v
    #             model.load_state_dict(ckpt,strict=False)

    #         # if self.cfgs.MODEL.FREEZE_DISP:
    #         #     for name, param in model.named_parameters():
    #         #         if name in pretrained_dict:
    #         #             param.requires_grad = False

    #     return model
