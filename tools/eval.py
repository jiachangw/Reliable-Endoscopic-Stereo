"""Evaluate using the released OpenStereo trainer and per-image metrics."""
import argparse
import json
import sys
from pathlib import Path

import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo.modeling import build_trainer
from stereo.utils import common_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--eval_data_cfg_file', required=True)
    parser.add_argument('--pretrained_model', required=True)
    parser.add_argument('--data_root')
    parser.add_argument('--split_file')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()
    args.run_mode = 'eval'
    args.dist_mode = False
    cfg = EasyDict(common_utils.config_loader(args.cfg_file))
    eval_cfg = EasyDict(common_utils.config_loader(args.eval_data_cfg_file))
    cfg.DATA_CONFIG = eval_cfg.DATA_CONFIG
    cfg.EVALUATOR = eval_cfg.EVALUATOR
    cfg.MODEL.PRETRAINED_MODEL = args.pretrained_model
    cfg.MODEL.STRICT_LOAD = True
    for info in cfg.DATA_CONFIG.DATA_INFOS:
        if args.data_root:
            info.DATA_PATH = args.data_root
        if args.split_file:
            info.DATA_SPLIT.EVALUATING = args.split_file
    torch.cuda.set_device(0)
    common_utils.set_random_seed(42)
    logger = common_utils.create_logger()
    trainer = build_trainer(args, cfg, 0, 0, logger, None)
    metrics = trainer.evaluate(0)
    result = dict(
        checkpoint=args.pretrained_model, config=args.cfg_file,
        eval_config=args.eval_data_cfg_file, num_pairs=len(trainer.eval_set),
        aggregation='mean of per-image metrics',
        metrics={key: float(value) for key, value in metrics.items()},
    )
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
