"""Predict left-view disparity for a rectified pair or a two-column pair list."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling import build_model
from stereo.utils import common_utils
from stereo.utils.disp_color import disp_to_color


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--pretrained_model', required=True)
    parser.add_argument('--left_img_path')
    parser.add_argument('--right_img_path')
    parser.add_argument('--pairs_file', help='two whitespace-separated paths per line')
    parser.add_argument('--data_root', default='.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true', help='enable model-specific mixed precision')
    return parser.parse_args()


def read_pairs(args):
    if args.pairs_file:
        if args.left_img_path or args.right_img_path:
            raise ValueError('Use either --pairs_file or a single pair')
        root = Path(args.data_root)
        with open(args.pairs_file, encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError('Each inference row must contain left right')
                yield root / fields[0], root / fields[1]
    elif args.left_img_path and args.right_img_path:
        yield Path(args.left_img_path), Path(args.right_img_path)
    else:
        raise ValueError('Provide --left_img_path and --right_img_path, or --pairs_file')


def prepare_pair(cfg, left, right):
    if left.shape != right.shape:
        raise ValueError('Left and right images must have the same shape')
    transforms = copy.deepcopy(cfg.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
    height, width = left.shape[:2]
    for transform in transforms:
        if transform.NAME == 'RightTopPad':
            transform.SIZE = [
                ((max(height, transform.SIZE[0]) + 31) // 32) * 32,
                ((max(width, transform.SIZE[1]) + 31) // 32) * 32,
            ]
    return build_transform_by_cfg(transforms)(dict(left=left, right=right))


@torch.no_grad()
def main():
    args = parse_args()
    cfg = EasyDict(common_utils.config_loader(args.cfg_file))
    common_utils.set_random_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = build_model(cfg.MODEL).to(device).eval()
    common_utils.load_params_from_file(
        model, args.pretrained_model, device, False, None, strict=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (left_path, right_path) in enumerate(read_pairs(args)):
        left = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32)
        right = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)
        height, width = left.shape[:2]
        sample = prepare_pair(cfg, left, right)
        sample = {key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
                  for key, value in sample.items()}
        dtype = torch.bfloat16 if cfg.MODEL.NAME == 'Monster' else torch.float16
        with torch.autocast(device_type=device.type, enabled=args.amp, dtype=dtype):
            if cfg.MODEL.NAME == 'Monster':
                output = model(sample, iters=cfg.MODEL.valid_iters)
            else:
                output = model(sample)
        disparity = output['disp_pred'][0, 0].float().cpu().numpy()
        # RightTopPad puts padding above and to the right of the original image.
        disparity = disparity[-height:, :width].astype(np.float32)
        stem = '{:06d}_{}'.format(index, left_path.stem)
        npy_path = out / (stem + '.npy')
        png_path = out / (stem + '.png')
        if npy_path.exists() or png_path.exists():
            raise FileExistsError('Use a new output directory: {}'.format(out))
        np.save(npy_path, disparity)
        max_disp = cfg.MODEL.get('MAX_DISP', cfg.MODEL.get('max_disp', 192))
        Image.fromarray(disp_to_color(disparity, max_disp=max_disp).astype(np.uint8)).save(png_path)
        records.append(dict(left=str(left_path), right=str(right_path),
                            disparity=npy_path.name, visualization=png_path.name,
                            height=height, width=width, unit='pixels'))
        print(npy_path)
    if not records:
        raise ValueError('No input pairs found')
    with (out / 'manifest.json').open('w', encoding='utf-8') as stream:
        json.dump(records, stream, indent=2)


if __name__ == '__main__':
    main()
