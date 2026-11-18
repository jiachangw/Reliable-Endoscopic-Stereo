# Reliable Endoscopic Stereo Matching

Model implementations, supervised student training, and stereo-pair inference
for MonSter, IGEV, and LightStereo.

## Installation

Use Linux with an NVIDIA GPU for training. A suggested
environment is Python 3.10 with the following matching PyTorch/CUDA wheels:

```bash
git clone https://github.com/jiachangw/Reliable-Endoscopic-Stereo.git
cd Reliable-Endoscopic-Stereo
conda create -n reliable-stereo python=3.10 -y
conda activate reliable-stereo
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

No model constructor downloads weights automatically.
W&B logging is disabled by default and does not require an account.

## Pretrained Models

The previously released weights remain available from
[Google Drive](https://drive.google.com/drive/folders/1qBsDXSet3cA3o7Ax8secUW1g9pcgboIM?usp=sharing).
Pass the downloaded file's actual path to `--pretrained_model`.

| Model | Config | Parameters | FPS |
| --- | --- | ---: | ---: |
| MonSter | [monster.yaml](cfgs/monster.yaml) | 388.69M | 3.01 |
| IGEV | [igev.yaml](cfgs/igev.yaml) | 12.60M | 4.44 |
| LightStereo | [lightstereo.yaml](cfgs/lightstereo.yaml) | 24.29M | 28.45 |

The parameter counts and FP32 FPS at 480 x 640, batch size 1, are the previously
reported measurements, not new measurements of this release.

The loader supports plain state dictionaries and `model_state`, `state_dict`, or
`model` wrappers, with or without the `module.` prefix. Loading is strict by
default: a wrong model or incomplete checkpoint raises an error.

## Inference

Inputs must be a rectified, same-sized RGB stereo pair. For example:

```bash
python tools/infer.py \
    --cfg_file cfgs/lightstereo.yaml \
    --pretrained_model checkpoints/lightstereo.pth \
    --left_img_path /path/to/left.png \
    --right_img_path /path/to/right.png \
    --output_dir output/pair_lightstereo
```

Use `cfgs/igev.yaml` or `cfgs/monster.yaml` with the corresponding checkpoint for
the other models. Inference uses FP32 by default; `--amp` enables mixed precision.
MonSter uses 32 refinement iterations. No separate monocular checkpoint is needed
when loading a complete MonSter checkpoint.

For multiple pairs, provide a text file with `left_path right_path` per row:

```bash
python tools/infer.py \
    --cfg_file cfgs/monster.yaml \
    --pretrained_model checkpoints/monster.pth \
    --pairs_file /path/to/pairs.txt --data_root /path/to/images \
    --output_dir output/pairs_monster
```

Each pair produces an unpadded float32 `.npy` disparity map in **pixels**, a
color `.png` visualization, and an entry in `manifest.json`. The PNG is not a
metric disparity map. Calibrated depth is `focal_length * baseline / disparity`,
with consistent focal-length, baseline, and disparity units.

## Training

Train with the configuration and data paths of the selected model:

```bash
torchrun --standalone --nproc_per_node=4 tools/train.py --dist_mode \
    --cfg_file xxx \
    --train_root xxx --train_list xxx \
    --eval_root xxx --eval_list xxx \
    --pretrained_model xxx \
    --extra_tag xxx --workers 8 --pin_memory
```

## Acknowledgments and Licenses

The implementations are adapted from [OpenStereo](https://github.com/XiandaGuo/OpenStereo),
[MonSter](https://github.com/Junda24/MonSter),
[IGEV](https://github.com/gangweiX/IGEV), and
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), including
DINOv2 components. Original author headers are retained.

Our original contributions are under [MIT](LICENSE).
