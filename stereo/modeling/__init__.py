"""Model families included in the endoscopic student release."""


def build_model(model_cfg):
    if model_cfg.NAME == 'LightStereo':
        from .models.lightstereo.lightstereo import LightStereo
        return LightStereo(model_cfg)
    if model_cfg.NAME == 'IGEV':
        from .models.igev.igev_stereo import IGEVStereo
        return IGEVStereo(model_cfg)
    if model_cfg.NAME == 'Monster':
        from .models.monster.core.monster import Monster
        return Monster(model_cfg)
    raise ValueError('Unsupported model: {}'.format(model_cfg.NAME))


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    if cfgs.MODEL.NAME == 'LightStereo':
        from .models.lightstereo.trainer import Trainer
    elif cfgs.MODEL.NAME == 'IGEV':
        from .models.igev.trainer import Trainer
    elif cfgs.MODEL.NAME == 'Monster':
        from .models.monster.trainer import Trainer
    else:
        raise ValueError('Unsupported model: {}'.format(cfgs.MODEL.NAME))
    return Trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)
