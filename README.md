# RadarMAE

Self-supervised Masked Autoencoder pretraining for radar Range-Azimuth (RA) maps,
with downstream object-detection fine-tuning on K-Radar.

## Upstream baselines

This codebase is built on top of two open-source baselines:

- **MAE pretraining** — adapted from
  [facebookresearch/mae](https://github.com/facebookresearch/mae)
  (`main_pretrain.py`, `engine_pretrain.py`, `models_radarmae.py`, `util/`).
- **K-Radar fine-tuning** — adapted from
  [kaist-avelab/K-Radar](https://github.com/kaist-avelab/K-Radar)
  (`finetune/`, including pipelines, datasets, models, utils, ops, and resources).

All modifications and new code (axial / stripe tokenizers, polar-BEV adapter,
distribution-matching losses, two-phase training pipeline, etc.) are layered on
top of these upstream sources. Per-file attribution headers are preserved
where the original authors provided them.

## Repository layout

```
RadarMAE/
├── main_pretrain.py            # MAE pretraining entry point
├── engine_pretrain.py                # Per-epoch training loop
├── models_radarmae.py                # RadarMAE encoder/decoder (axial / stripe / patch tokenizers)
│
├── util/                             # MAE shared utilities
│   ├── lr_sched.py
│   ├── misc.py
│   ├── pos_embed.py
│   └── visualization.py              # Reconstruction visualization
│
└── finetune/                         # K-Radar object-detection fine-tuning
    ├── main_train.py                 # Fine-tuning entry point
    ├── configs/
    │   ├── cfg_RadarMAE.yml  # Active config (RTNH-wide + MAE backbone)
    │   ├── config_general.py
    │   └── config_ui.py
    ├── pipelines/
    │   └── pipeline_detection_v1_0.py
    ├── datasets/                     # K-Radar dataset wrappers (v1.0–v2.1)
    ├── models/                       # RadarBase = MAE backbone + anchor head
    │   ├── backbone_2d/              # `mae_polar_bev.py` (RadarMAE encoder readout), `stripe_modules.py`
    │   ├── head/                     # `anchor_head.py` (AnchorHeadSingle) + target assigners
    │   ├── skeletons/                # `rdr_base.py` (RadarBase)
    │   └── model_utils/              # NMS utilities
    ├── ops/                          # CUDA extensions (iou3d_nms, roiaware_pool3d, bev_pool, pointnet2)
    │                                 # Source only — rebuild instructions below.
    ├── utils/
    │   ├── pretrain_inspect.py       # Reads MAE ckpt args, drives backbone overrides
    │   ├── kitti_eval/, Rotated_IoU/ # Detection-metric utilities
    │   ├── box_coder_utils.py, box_utils.py, common_utils.py, loss_utils.py, …
    └── resources/                    # Calibration matrices and split lists
        └── split/                    # train.txt / val.txt / test.txt and 1-20 / origin variants
```

Fine-tuning uses the original K-Radar labels (`<seq>/info_label/*.txt`). The
revised K-Radar labels (v1.1 / v2.0 / v2.1) are not included; get them from the
[K-Radar repository](https://github.com/kaist-avelab/K-Radar) if needed.

## Installation

```bash
git clone https://github.com/DGIST-IRSLab/RadarMAE.git
cd RadarMAE
```

## Environment

Tested with Python 3.10, PyTorch 2.7.1 and CUDA 12.8. Install PyTorch for your
CUDA version first, then the remaining packages:

```bash
conda create -n radarmae python=3.10 -y && conda activate radarmae
conda install -c conda-forge libusb -y   # system library needed by open3d (or: apt install libusb-1.0-0)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`timm==0.6.12` is required exactly (checked at startup of `main_pretrain.py`).
`wandb` is optional.

## Building the CUDA ops (K-Radar fine-tuning)

Fine-tuning needs two CUDA extensions from `finetune/ops/` (`iou3d_nms`,
`roiaware_pool3d`), compiled in place. This requires a CUDA toolkit (`nvcc`)
with the same version as your PyTorch build (12.8 above):

```bash
conda install -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 -y   # skip if a matching toolkit is installed
cd finetune
CUDA_HOME=$CONDA_PREFIX python setup.py build_ext --inplace           # or CUDA_HOME=/usr/local/cuda-12.8
```

## Radar input: from 4DRT to RA map

K-Radar provides each radar frame as a 4D radar tensor (4DRT,
`<seq>/radar_tesseract/tesseract_XXXXX.mat`, key `arrDREA`): power measurements
over Doppler × range × elevation × azimuth = 64 × 256 × 37 × 107 bins
(range 0–118 m, azimuth −53°…53°, elevation −18°…18°; bin values in
`finetune/resources/info_arr.mat`).

RadarMAE does not use the Doppler and elevation dimensions directly; it works on
a 2D range–azimuth (RA) power map:

1. **Doppler removed.** Power is averaged over the Doppler bins (bin 0 excluded,
   as in K-Radar's `get_cube_polar`) and scaled by 1e-13, giving a 256 × 107 × 37
   range–azimuth–elevation power cube (`polar3d_XXXXX_rae.npy`). The Doppler
   (radial-velocity) channel is discarded.
2. **Elevation mean.** The cube is averaged along elevation, giving the RA map,
   stored as 107 × 256 (azimuth × range) in `polar3d_XXXXX_rae_ra.npy`.

Both pretraining and fine-tuning read these `*_rae_ra.npy` RA maps.

## Pretraining

```bash
python main_pretrain.py --help
```

The default reads K-Radar RA-map NPYs from `--data_path` and uses the splits in
`finetune/resources/split/`. A precomputed cache (`--cache_dir`) of normalized
RA tensors is supported for faster I/O.

## Pretrained checkpoint

The K-Radar-pretrained RadarMAE checkpoint (614 MB) is distributed through
[GitHub Releases](https://github.com/DGIST-IRSLab/RadarMAE/releases). Download it
to the repository root:

```bash
wget https://github.com/DGIST-IRSLab/RadarMAE/releases/download/v1.0/RadarMAE.pth
sha256sum RadarMAE.pth   # 800c2d86d3ae7f2d4741db87223156f0922549f2293096df2bddffce1ee8b6e1
```

It holds the full encoder/decoder weights (`model`) and the pretraining
arguments (`args`, an `argparse.Namespace`; load with `weights_only=False`).
Fine-tuning reads `args` to configure the encoder.

## K-Radar fine-tuning

The active config is `finetune/configs/cfg_RadarMAE.yml`. Its
`MODEL.BACKBONE.MAE.PRETRAIN_PATH` points to `RadarMAE.pth` in the repository
root (override with `--pretrain-path`), then:

```bash
cd finetune
python main_train.py --cfg configs/cfg_RadarMAE.yml
```

`main_train.py` enforces K-Radar v1.0 `info_label/*.txt` labels at
launch (`prepare_v1_0_label_cfg`).

## Data and outputs layout

All defaults point to project-relative paths so the release runs without
hardcoded environment paths.

```
RadarMAE/
├── RadarMAE.pth                      # Released pretrained checkpoint (download separately)
├── data/                             # Datasets (place or symlink your data here)
│   └── K-Radar_RA/
│       ├── <seq>/polar3d_XXXXX_rae_ra.npy   # RA maps (input of both stages)
│       └── RAE/<seq>/info_label/, info_calib/, description.txt   # K-Radar labels / calibration
└── outputs/                          # Created at run time
    ├── pretrain/                     # MAE pretraining checkpoints + logs (+ wandb/)
    ├── finetune/                     # Fine-tuning experiments (+ logs/, wandb_logs/, effective_configs/)
    └── cache/preprocessed_ra/        # Optional precache of normalized RA tensors
```

Either place the data directly in `data/` or symlink your local copy:

```bash
mkdir -p data && ln -s /path/to/K-Radar_RA data/K-Radar_RA
```

## Notes

- Path defaults resolve relative to the repository: `main_pretrain.py` uses the
  project root (`--data_path`, `--output_dir` override them), and paths inside
  `finetune/configs/*.yml` are relative to `finetune/`.
- No GPU is pinned in code; select one with `CUDA_VISIBLE_DEVICES`.
- Compiled CUDA artifacts (`*.so`), `__pycache__/`, prior experiment outputs,
  and figure assets have been removed from this release. Rebuild ops as
  described above.

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

The fine-tuning code under `finetune/` is adapted from
[kaist-avelab/K-Radar](https://github.com/kaist-avelab/K-Radar) and keeps its
original license: the `K-Radar` dataset is published under the CC BY-NC-ND
License, and all K-Radar codes are published under the Apache License 2.0.
