# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import json
import shutil
from pathlib import Path
import tqdm
from easydict import EasyDict
import torch
import torch.distributed as dist

# from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo.utils import common_utils
from stereo.modeling import build_trainer


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # dataloader
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    parser.add_argument('--base_seed', '--seed', type=int, default=42, help='base random seed')
    parser.add_argument('--train_root')
    parser.add_argument('--eval_root')
    parser.add_argument('--train_list')
    parser.add_argument('--eval_list')
    parser.add_argument('--pretrained_model')
    parser.add_argument('--mono_pretrained')
    parser.add_argument('--wandb', action='store_true', help='enable optional online logging')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    cfgs.DATA_CONFIG.SEED = args.base_seed
    if args.pretrained_model:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    if args.mono_pretrained:
        cfgs.MODEL.MONO_PRETRAINED = args.mono_pretrained
    if (cfgs.MODEL.NAME == 'Monster' and not cfgs.MODEL.PRETRAINED_MODEL
            and not cfgs.MODEL.get('MONO_PRETRAINED') and cfgs.MODEL.CKPT < 0):
        raise ValueError('MonSter needs --mono_pretrained or a full --pretrained_model checkpoint')
    for info in cfgs.DATA_CONFIG.DATA_INFOS:
        if args.train_root:
            info.DATA_PATH = args.train_root
        if args.eval_root:
            info.EVAL_DATA_PATH = args.eval_root
        if args.train_list:
            info.DATA_SPLIT.TRAINING = args.train_list
        if args.eval_list:
            info.DATA_SPLIT.EVALUATING = args.eval_list

    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = list(set(dataset_names))
    if len(unique_dataset_names) == 1:
        exp_dataset_dir = unique_dataset_names[0]
    else:
        exp_dataset_dir = 'MultiDataset'
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    args.run_mode = 'train'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        group_rank = int(os.environ["GROUP_RANK"])
    else:
        local_rank = 0
        global_rank = 0
        group_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    seed = args.base_seed + global_rank
    common_utils.set_random_seed(seed=seed)

    # savedir
    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))
    if os.path.exists(args.output_dir) and args.extra_tag != 'debug' and cfgs.MODEL.CKPT == -1:
        raise Exception('There is already an exp with this name')
    if args.dist_mode:
        dist.barrier()
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if not os.path.exists(args.ckpt_dir) and global_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if global_rank == 0:
        common_utils.backup_source_code(os.path.join(args.output_dir, 'code'))
    if args.dist_mode:
        dist.barrier()

    # logger
    log_file = os.path.join(args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    # wandb
    if global_rank == 0 and args.wandb:
        import wandb
        wandb_init_args = dict(
            project=os.environ.get('SWANLAB_PROJECT', 'stereo-matching'),
            name=f"{args.tag}_{args.extra_tag}",
            config=dict(cfgs),
            dir=args.output_dir
        )
        wandb_init_args['mode'] = os.environ.get('WANDB_MODE', 'online')
        wandb_logger = wandb.init(**wandb_init_args)
        print('init done')
        # 记录所有参数
        wandb_logger.config.update(args)
    else:
        wandb_logger = None
    # tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard')) if global_rank == 0 else None
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)
    if global_rank == 0:
        shutil.copy2(args.cfg_file, args.output_dir)
        with open(os.path.join(args.output_dir, 'resolved_config.json'), 'w') as fp:
            json.dump(cfgs, fp, indent=2)
        resumed = cfgs.MODEL.CKPT >= 0
        resume_path = (os.path.join(args.ckpt_dir, 'checkpoint_epoch_%d.pth' % cfgs.MODEL.CKPT)
                       if resumed else None)
        with open(os.path.join(args.output_dir, 'run_manifest.json'), 'w') as fp:
            json.dump(dict(
                seed=args.base_seed, source_checkpoint=cfgs.MODEL.PRETRAINED_MODEL or None,
                resume_checkpoint=resume_path, resumed=resumed,
                frozen_mono_checkpoint=cfgs.MODEL.get('MONO_PRETRAINED'),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
            ), fp, indent=2)

    # trainer
    model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, wandb_logger)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        if current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == model_trainer.total_epochs - 1:
            model_trainer.evaluate(current_epoch)
    if wandb_logger is not None:
        wandb_logger.finish()
    if args.dist_mode:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
