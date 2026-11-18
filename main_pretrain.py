# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# RadarMAE pretraining entry point.
# Adapted from facebookresearch/mae (https://github.com/facebookresearch/mae).
# Modifications: axial / stripe tokenizers, RA-map dataset and split handling,
# physical positional encoding, distribution-matching losses, etc.
# --------------------------------------------------------
import os
import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from scipy.io import loadmat

import timm

if timm.__version__ != "0.6.12":
    raise RuntimeError(
        f"Expected timm==0.6.12, got timm=={timm.__version__}. "
        "Install with: pip install -U 'timm==0.6.12'"
    )
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

import models_radarmae

from engine_pretrain import train_one_epoch
from util.visualization import save_reconstruction_plot

TRAIN_SPLIT_FILENAME = "train_origin.txt"
VAL_SPLIT_FILENAME = "test_origin.txt"

DEFAULT_SPLIT_DIR = Path(__file__).resolve().parent / "finetune" / "resources" / "split"
DEFAULT_TRAIN_SPLIT = str(DEFAULT_SPLIT_DIR / TRAIN_SPLIT_FILENAME)
DEFAULT_VAL_SPLIT = str(DEFAULT_SPLIT_DIR / VAL_SPLIT_FILENAME)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_KRADAR_DATA_PATH = str(PROJECT_ROOT / "data" / "K-Radar_RA")
DEFAULT_CACHE_DIR = str(PROJECT_ROOT / "outputs" / "cache" / "preprocessed_ra")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "pretrain")


def _import_wandb():
    """
    Import the real Weights & Biases package even if this repo contains a local 'wandb/' folder.
    Returns None if wandb is not installed.
    """
    import importlib.machinery
    import importlib.util
    import sys

    script_dir = str(Path(__file__).resolve().parent)
    search_paths = [p for p in sys.path if p and p != script_dir]
    spec = importlib.machinery.PathFinder.find_spec("wandb", search_paths)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "init"):
        return None
    return module


class RadarNpyDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_list: List[str],
        transform=None,
        range_limits: Tuple[float, float] = (0.0, 118.0),
        azimuth_limits: Tuple[float, float] = (-53.0, 53.0),
        preprocess: str = "db",
        log_scale: Optional[bool] = None,
        ra_map_norm_enable: bool = True,
        ra_map_norm_mean: float = None,  
        ra_map_norm_std: float = None,  
        pretrain_axis: str = "azimuth_range",
        range_bins: Optional[int] = None,
        azimuth_bins: Optional[int] = None,
        cache_dir: Optional[str] = None,
        return_target: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.rel_paths = [Path(f) for f in file_list]
        self.files = [self.root_dir / f for f in self.rel_paths]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None and not self.cache_dir.exists():
            raise FileNotFoundError(f"cache_dir does not exist: {self.cache_dir}")
        if len(self.files) == 0:
            raise FileNotFoundError(f"No NPY files were provided under '{root_dir}'.")
        self.range_limits = range_limits
        self.azimuth_limits = azimuth_limits
        if log_scale is not None:
            preprocess = "log1p" if log_scale else "none"
        self.preprocess = None if preprocess is None else str(preprocess).lower()
        self.ra_map_norm_enable = bool(ra_map_norm_enable)
        if ra_map_norm_mean is None or ra_map_norm_std is None:
            raise ValueError(
                "ra_map_norm_mean and ra_map_norm_std must be explicitly provided. "
                "These values are controlled via argparse --ra_map_norm_mean and --ra_map_norm_std."
            )
        self.ra_map_norm_mean = float(ra_map_norm_mean)
        self.ra_map_norm_std = float(ra_map_norm_std)
        if abs(self.ra_map_norm_std) < 1.0e-12:
            raise ValueError("RA map normalization std must be non-zero.")
        self.pretrain_axis = str(pretrain_axis).lower().strip()
        if self.pretrain_axis not in ("azimuth_range", "range_azimuth"):
            raise ValueError(
                f"Unsupported pretrain_axis='{pretrain_axis}'. "
                "Use 'azimuth_range' or 'range_azimuth'."
            )
        self.range_bins = int(range_bins) if range_bins is not None else None
        self.azimuth_bins = int(azimuth_bins) if azimuth_bins is not None else None
        self.return_target = bool(return_target)
        self.target_shape = (
            (self.range_bins, self.azimuth_bins)
            if (self.range_bins is not None and self.azimuth_bins is not None)
            else None
        )

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _ra_map_to_db(x: np.ndarray) -> np.ndarray:
        x = np.abs(np.asarray(x, dtype=np.float32))
        eps = 1.0e-12
        x_db = 20.0 * np.log10(x + eps)
        return x_db.astype(np.float32, copy=False)

    def _normalize_ra_map(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if not self.ra_map_norm_enable:
            return x
        x_norm = (x - self.ra_map_norm_mean) / self.ra_map_norm_std
        return x_norm.astype(np.float32, copy=False)

    @staticmethod
    def _reduce_to_2d(ra_arr: np.ndarray) -> Optional[np.ndarray]:
        if ra_arr.ndim == 2:
            return ra_arr.astype(np.float32)
        if ra_arr.ndim == 3:
            return ra_arr.sum(axis=0).astype(np.float32)
        if ra_arr.ndim == 4:
            return ra_arr.sum(axis=(0, 3)).astype(np.float32)
        return None

    def _orient_to_range_az(self, ra_arr: np.ndarray) -> np.ndarray:
        if self.target_shape is None:
            return ra_arr
        if ra_arr.shape == self.target_shape:
            return ra_arr
        if ra_arr.shape == (self.target_shape[1], self.target_shape[0]):
            return ra_arr.transpose(1, 0)
        return ra_arr

    def _format_output(self, tensor: torch.Tensor):
        if self.return_target:
            return tensor, tensor
        return tensor

    def __getitem__(self, idx):
        if self.cache_dir is not None:
            cache_path = self.cache_dir / self.rel_paths[idx]
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cached file missing: {cache_path}. Re-run precache_dataset.py "
                    "or pass --no_cache to use raw preprocessing."
                )
            arr = np.load(str(cache_path))
            tensor = torch.from_numpy(np.ascontiguousarray(arr))
            if self.transform is not None:
                tensor = self.transform(tensor)
            return self._format_output(tensor)

        path = self.files[idx]
        ra = np.load(path).astype(np.float32)
        ra = np.squeeze(ra)
        ra = self._reduce_to_2d(ra)
        if ra is None:
            raise ValueError(f"Unsupported RA shape at {path}: {tuple(np.asarray(np.load(path)).shape)}")

        ra = self._orient_to_range_az(ra)
        if self.target_shape is not None and ra.shape != self.target_shape:
            raise ValueError(
                f"RA shape mismatch at {path}: got {ra.shape}, expected (range, azimuth)={self.target_shape}"
            )

        if self.preprocess == "db":
            # Reference KRadar v13 preprocessing
            ra = self._ra_map_to_db(ra)
        elif self.preprocess == "log1p":
            ra = np.log1p(np.abs(ra))
        elif self.preprocess in ("none", None):
            pass
        else:
            raise ValueError(
                f"Unknown preprocess='{self.preprocess}'. Use 'db', 'log1p', or 'none'."
            )
        ra = self._normalize_ra_map(ra)

        # Match the RadarMAE pretraining axis convention when requested.
        if self.pretrain_axis == "azimuth_range":
            ra = ra.transpose(1, 0)

        data = ra
        if self.transform is not None:
            tensor = self.transform(data)
        else:
            tensor = torch.from_numpy(data).unsqueeze(0)
        return self._format_output(tensor)


def _default_info_arr_path() -> Path:
    env_path = os.environ.get("INFO_ARR_PATH") or os.environ.get("K_RADAR_INFO_ARR_PATH")
    if env_path:
        cand = Path(env_path).expanduser()
        if cand.exists():
            return cand
        raise FileNotFoundError(f"INFO_ARR_PATH is set but file does not exist: {cand}")

    cand = PROJECT_ROOT / "finetune" / "resources" / "info_arr.mat"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        "info_arr.mat not found. "
        "Set --info_arr_path or env INFO_ARR_PATH. "
        f"Tried: {cand}"
    )


def _load_sensor_bins(info_arr_path: Path):
    vals = loadmat(str(info_arr_path))
    arr_range = vals["arrRange"].reshape(-1).astype(np.float32)
    arr_azimuth = vals["arrAzimuth"].reshape(-1).astype(np.float32) * np.pi / 180.0
    return arr_range, arr_azimuth


def _load_sensor_bins_for_args(args):
    info_arr_path = Path(args.info_arr_path) if args.info_arr_path else _default_info_arr_path()
    return _load_sensor_bins(info_arr_path)


def _apply_dataset_preset(args):
    if getattr(args, "npy_glob", None) is None:
        args.npy_glob = "*_rae_ra.npy"


def _resolve_data_root(preferred: str) -> Path:
    cand = Path(preferred or DEFAULT_KRADAR_DATA_PATH)
    if cand.exists():
        return cand
    raise FileNotFoundError(f"K-Radar RA-map data not found: {cand} (set --data_path)")


def _read_split_file(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def _kradar_split_to_npy_list(split_path, data_root):
    lines = _read_split_file(split_path)
    results = []
    missing = 0
    for ln in lines:
        if "," not in ln:
            continue
        seq, label_file = ln.split(",", 1)
        frame_id = Path(label_file).stem.split("_")[0]
        seq_dir = Path(data_root) / Path(seq)
        candidate = seq_dir / f"polar3d_{frame_id}_rae_ra.npy"
        if candidate.exists():
            results.append(str(candidate.relative_to(data_root)))
            continue
        fallback = sorted(seq_dir.glob(f"*{frame_id}_rae_ra.npy")) if seq_dir.exists() else []
        if not fallback and seq_dir.exists():
            fallback = sorted(seq_dir.rglob(f"*{frame_id}_rae_ra.npy"))
        if fallback:
            results.append(str(fallback[0].relative_to(data_root)))
        else:
            missing += 1
    if missing > 0 and misc.is_main_process():
        print(f"[split] {missing} entries from {split_path} were skipped (missing npy)")
    return results


def build_train_val_split(args):
    data_root = _resolve_data_root(args.data_path)
    args.data_path = str(data_root)

    def resolve_list(path):
        p = Path(path)
        if not p.is_absolute():
            candidates = [
                Path(path),
                Path(__file__).resolve().parent / p,
                Path(__file__).resolve().parent.parent / p,
                Path(args.data_path) / p,
            ]
            for cand in candidates:
                if cand.exists():
                    p = cand
                    break
            else:
                p = Path(args.data_path) / p

        lines = _read_split_file(p)
        # Support K-Radar split format: "seq,label_txt" -> map to "*_rae_ra.npy" paths.
        if any("," in ln for ln in lines[:100]):
            return _kradar_split_to_npy_list(p, data_root)
        return lines

    if args.train_list and args.val_list:
        train_list = resolve_list(args.train_list)
        val_list = resolve_list(args.val_list)
    elif args.train_list or args.val_list:
        raise ValueError("train_list and val_list must be specified together.")
    else:
        tr = DEFAULT_SPLIT_DIR / TRAIN_SPLIT_FILENAME
        va = DEFAULT_SPLIT_DIR / VAL_SPLIT_FILENAME
        if not (tr.exists() and va.exists()):
            raise FileNotFoundError(
                f"Default split files not found: {tr}, {va}. "
                "Either provide --train_list and --val_list explicitly, "
                f"or place split files at {DEFAULT_SPLIT_DIR}."
            )
        train_list = _kradar_split_to_npy_list(tr, data_root)
        val_list = _kradar_split_to_npy_list(va, data_root)
        if misc.is_main_process():
            print(f"[split] Using project split files: train={len(train_list)}, val={len(val_list)}")

    if len(train_list) == 0 or len(val_list) == 0:
        raise RuntimeError(
            "Split files produced empty train/val lists. "
            f"train_list={args.train_list}, val_list={args.val_list}"
        )
    return train_list, val_list


def _serialize_arg_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _save_training_args(args) -> None:
    arg_keys = getattr(args, "_parser_argument_keys", None)
    if arg_keys is None:
        arg_keys = sorted(key for key in vars(args).keys() if not key.startswith("_"))
    else:
        arg_keys = sorted({key for key in arg_keys if hasattr(args, key)})

    target_dirs = []
    for directory in (getattr(args, "output_dir", None), getattr(args, "log_dir", None)):
        if not directory:
            continue
        path = Path(directory)
        if path not in target_dirs:
            target_dirs.append(path)

    if not target_dirs:
        return

    lines = [
        f"saved_at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"command: {' '.join(sys.argv)}",
        "",
    ]
    for key in arg_keys:
        lines.append(f"{key}: {_serialize_arg_value(getattr(args, key))}")
    text = "\n".join(lines) + "\n"

    saved_paths = []
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "training_args.txt"
        output_path.write_text(text, encoding="utf-8")
        saved_paths.append(str(output_path))

    if misc.is_main_process():
        print(f"[config] Saved training arguments: {', '.join(saved_paths)}")


def _parse_patch_size(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"patch_size must have 1 or 2 values, got {value}")
    text = str(value).strip().lower()
    if "x" in text:
        parts = text.split("x")
    elif "," in text:
        parts = text.split(",")
    else:
        return int(text)
    if len(parts) != 2:
        raise ValueError(f"patch_size must be int or HxW, got '{value}'")
    return (int(parts[0]), int(parts[1]))


def _parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def get_args_parser():
    parser = argparse.ArgumentParser("RadarMAE pre-training", add_help=False)
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, help="Accumulate gradient iterations")

    parser.add_argument("--model", default="mae_vit_base_patch8", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--in_chans", default=1, type=int, help="Number of input channels")
    parser.add_argument(
        "--patch_size",
        default="4x16",
        type=str,
        help="Patch size override (int or HxW, e.g., 16 or 4x16).",
    )
    parser.add_argument(
        "--stripe_patch_width",
        default="2x3",
        type=str,
        help="Axial stripe patch width override (int or HxW, e.g., 3 or 2x3).",
    )
    parser.add_argument(
        "--stripe_patch_stride",
        default="2x3",
        type=str,
        help="Axial stripe patch stride override (int or HxW, e.g., 1 or 1x1).",
    )
    parser.add_argument(
        "--tokenization_mode",
        default="axial",
        type=str,
        choices=["axial", "patch"],
        help=(
            "Encoder tokenization mode. 'axial' uses stripe tokens along both axes "
            "(default). 'patch' uses standard 2D patch embedding like original MAE."
        ),
    )
    parser.add_argument(
        "--pos_embed_mode",
        default="ppe",
        type=str,
        choices=["ppe", "pe"],
        help=(
            "Positional embedding mode for the encoder. "
            "'ppe' uses PhysicalPositionalEncoding (Fourier + MLP, learnable). "
            "'pe' uses fixed sine-cosine positional embedding. "
            "Only affects axial tokenization mode; patch mode always uses fixed 2D sincos."
        ),
    )
    parser.add_argument(
        "--conv_stem_mode",
        default="per_token",
        type=str,
        choices=["global", "per_token"],
        help=(
            "Conv stem application mode (axial tokenization only). "
            "'global' applies conv_stem to the entire image before stripe tokenization (default). "
            "'per_token' applies conv_stem independently to each stripe, eliminating inter-token information leakage."
        ),
    )
    parser.add_argument(
        "--no_input_norm",
        action="store_true",
        help="Disable the v13-style BatchNorm2d input normalization before ConvStem.",
    )

    parser.add_argument("--input_size", default=256, type=int, help="Fallback square input size")
    parser.add_argument("--input_height", default=108, type=int, help="Input height (overrides input_size when used with width)")
    parser.add_argument("--input_width", default=258, type=int, help="Input width (overrides input_size when used with height)")

    parser.add_argument("--mask_ratio", default=0.75, type=float, help="Masking ratio")
    parser.add_argument(
        "--masking_mode",
        default="weighted_block_masking",
        type=str,
        choices=["random", "weighted_masking", "weighted_block_masking", "random_block_masking"],
        help=(
            "Token masking mode. 'weighted_masking' masks brighter axial tokens more often while keeping a small "
            "visible hint among the top-intensity tokens. 'weighted_block_masking' samples 3-5 token blocks on "
            "both range and azimuth axes from intensity-weighted start indices and leaves a single-axis hint token. "
            "Under 'patch' tokenization, 'weighted_block_masking' masks square patch blocks spreading both ways from "
            "an intensity-sampled centre patch, leaving one random hint patch per block. "
            "'random_block_masking' is the same block procedure with uniformly sampled centres — it isolates the "
            "block structure from the intensity weighting."
        ),
    )
    parser.add_argument(
        "--patch_block_min_size",
        default=3,
        type=int,
        help="Min square block side (in patches) for patch-mode weighted block masking. Odd sizes only.",
    )
    parser.add_argument(
        "--patch_block_max_size",
        default=5,
        type=int,
        help="Max square block side (in patches) for patch-mode weighted block masking. Odd sizes only.",
    )
    parser.add_argument(
        "--patch_block_max_steps",
        default=40,
        type=int,
        help="Block placement steps for patch-mode weighted block masking before budget alignment.",
    )
    parser.add_argument("--norm_pix_loss", action="store_true", help="Use normalized pixels as targets for computing loss")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument(
        "--recon_loss",
        default="weighted_l1",
        type=str,
        choices=["mse", "l1", "weighted_mse", "weighted_l1"],
        help=(
            "Reconstruction loss type for RadarMAE training. "
            "'weighted_mse' and 'weighted_l1' are explicit aliases for the base loss "
            "with patch-intensity weighting."
        ),
    )
    parser.add_argument(
        "--loss_weighting",
        default="patch_intensity",
        type=str,
        choices=["none", "patch_intensity"],
        help="Loss weighting mode. 'patch_intensity' upweights brighter target patches.",
    )
    parser.add_argument(
        "--enable_dist_loss",
        default=True,
        type=lambda v: v.lower() not in ("false", "0", "no", "off") if isinstance(v, str) else bool(v),
        help="Enable axis-specific distribution matching losses (default: True). Use --enable_dist_loss false to disable.",
    )
    parser.add_argument(
        "--lambda_multipath",
        default=0.1,
        type=float,
        help="Weight for the multipath_loss (§4.3.1) on range-axis masked pixels.",
    )
    parser.add_argument(
        "--lambda_sidelobs",
        default=0.1,
        type=float,
        help="Weight for the sidelobs_loss (§4.3.2) on azimuth-axis masked pixels.",
    )
    parser.add_argument(
        "--dist_loss_axis_routing",
        default="physical",
        type=str,
        choices=["physical", "swapped"],
        help=(
            "Axis routing for distribution losses. 'physical' applies multipath/range loss "
            "to range-masked pixels and sidelobe/azimuth loss to azimuth-masked pixels. "
            "'swapped' applies them to the opposite axes."
        ),
    )

    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm.')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate")
    parser.add_argument("--blr", type=float, default=1e-5, metavar="LR", help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound")
    parser.add_argument("--warmup_epochs", type=int, default=50, metavar="N", help="epochs to warmup LR")

    parser.add_argument("--data_path", default=DEFAULT_KRADAR_DATA_PATH, type=str, help="dataset path")
    parser.add_argument(
        "--cache_dir",
        default=DEFAULT_CACHE_DIR,
        type=str,
        help=(
            "Path to preprocessed cache produced by precache_dataset.py. "
            "When enabled, dataset reads cached (1, H, W) tensors and skips "
            "raw dB/normalize/orient/resize per-step. Pass empty string or "
            "--no_cache to force raw preprocessing."
        ),
    )
    parser.add_argument("--use_cache", action="store_true", help="Enable preprocessed cache.")
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", help="Disable preprocessed cache.")
    parser.set_defaults(use_cache=False)
    parser.add_argument(
        "--train_list",
        default=DEFAULT_TRAIN_SPLIT,
        type=str,
        help=f"Path to train split txt (default: {DEFAULT_TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--val_list",
        default=DEFAULT_VAL_SPLIT,
        type=str,
        help=f"Path to val/test split txt (default: {DEFAULT_VAL_SPLIT})",
    )
    parser.add_argument(
        "--info_arr_path",
        default=str(PROJECT_ROOT / "finetune" / "resources" / "info_arr.mat"),
        type=str,
        help="Path to info_arr.mat",
    )
    parser.add_argument(
        "--npy_glob",
        default=None,
        type=str,
        help="Pattern used for auto-discovering NPY files. Defaults to '*_rae_ra.npy'.",
    )
    parser.add_argument(
        "--preprocess",
        default="db",
        type=str,
        choices=["db", "log1p", "none"],
        help="Input preprocessing mode (reference default: db).",
    )
    parser.add_argument(
        "--ra_map_norm_enable",
        action="store_true",
        help="Enable fixed RA-map normalization (reference config).",
    )
    parser.add_argument(
        "--ra_map_norm_disable",
        action="store_false",
        dest="ra_map_norm_enable",
        help="Disable fixed RA-map normalization.",
    )
    parser.set_defaults(ra_map_norm_enable=True)
    parser.add_argument(
        "--ra_map_norm_mean",
        default=-42.3409423828,  # entire, with val set
        # default=-42.3303226670, # entire, no val set
        #default=-41.2429644302, #_k-radar_20
        type=float,
        help="Reference RA-map normalization mean after dB conversion.",
    )
    parser.add_argument(
        "--ra_map_norm_std",
        default=8.3755702972, # entire, with val set
        #default=8.3878404246, # entire, no val set
        #default=9.0808973724, # _k-radar_20
        type=float,
        help="Reference RA-map normalization std after dB conversion.",
    )
    parser.add_argument(
        "--pretrain_axis",
        default="azimuth_range",
        type=str,
        choices=["azimuth_range", "range_azimuth"],
        help="Axis convention for RadarMAE pretraining input tensor.",
    )
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="Disable train-only z-score standardization.",
    )
    parser.add_argument(
        "--use_train_standardize",
        action="store_false",
        dest="no_standardize",
        help="Enable train-only z-score standardization (legacy option).",
    )
    parser.set_defaults(no_standardize=True)
    parser.add_argument(
        "--standardize_stats_path",
        default=None,
        type=str,
        help="Optional path to save/load train mean/std stats (.npz). Defaults to <output_dir>/train_standardize_stats.npz",
    )
    parser.add_argument(
        "--standardize_recompute",
        action="store_true",
        help="Recompute train mean/std even if stats file exists.",
    )
    parser.add_argument(
        "--standardize_eps",
        default=1e-6,
        type=float,
        help="Minimum std clamp for standardization (avoid divide-by-zero).",
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="path where to save",
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument(
        "--amp_dtype",
        default="fp16",
        type=str,
        choices=["fp16", "bf16"],
        help="Autocast dtype for mixed precision. Use bf16 on Hopper/Ampere GPUs for extra throughput.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile for ~1.3-1.5x speedup on H100/H200.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    parser.add_argument(
        "--ckpt_interval",
        default=30,
        type=int,
        help="Save a numbered checkpoint every N epochs (best/last are always saved).",
    )
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--prefetch_factor",
        default=4,
        type=int,
        help="Number of prefetched batches per worker when num_workers > 0.",
    )
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--loss_curve_interval",
        default=10,
        type=int,
        help="Save train loss curve every N epochs. Set <= 0 to disable periodic updates.",
    )

    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--no_wandb",
        action="store_false",
        dest="use_wandb",
        help="Disable wandb logging.",
    )
    parser.add_argument("--wandb_project", default="RadarMAE", type=str, help="wandb project name")
    parser.add_argument(
        "--wandb_mode",
        default="offline",
        type=str,
        choices=["online", "offline", "disabled"],
        help="wandb mode (offline recommended on clusters; use 'wandb sync' later)",
    )
    parser.add_argument(
        "--wandb_dir",
        default=str(PROJECT_ROOT / "outputs" / "pretrain" / "wandb"),
        type=str,
        help="Directory to store offline wandb logs",
    )
    parser.add_argument("--wandb_name", default=None, type=str, help="wandb run name (default auto)")
    parser.set_defaults(use_wandb=False)
    parser.add_argument(
        "--profile_runtime",
        action="store_true",
        help="Emit step-level and epoch-level runtime breakdowns to stdout for Slurm log inspection.",
    )
    parser.add_argument(
        "--profile_steps",
        default=5,
        type=int,
        help="Number of iterations per profiled epoch to emit detailed runtime breakdowns for.",
    )
    parser.add_argument(
        "--profile_epochs",
        default=1,
        type=int,
        help="Number of initial epochs to profile when --profile_runtime is enabled.",
    )
    return parser


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _measure_flops(model, device, input_size, in_chans, mask_ratio):
    """Measure GFLOPs for one forward pass of RadarMAE (full: encoder w/ masking + decoder + loss)
    and for the encoder alone (visible tokens only, batch=1).
    Uses torch.utils.flop_counter.FlopCounterMode (PyTorch 2.x).

    Note: FlopCounterMode uses the module tracker which registers grad hooks,
    so we must run under grad-enabled mode (not torch.no_grad())."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    h, w = input_size
    dummy = torch.randn(1, in_chans, h, w, device=device)

    total_flops = None
    enc_flops = None
    try:
        fc_total = FlopCounterMode(display=False)
        with fc_total:
            model(dummy, mask_ratio=mask_ratio)
        total_flops = fc_total.get_total_flops()

        fc_enc = FlopCounterMode(display=False)
        with fc_enc:
            enc_out = model.forward_encoder(dummy, mask_ratio)
        enc_flops = fc_enc.get_total_flops()
        del enc_out
    except Exception as e:
        print(f"[flops] measurement failed: {type(e).__name__}: {e}")
    finally:
        if was_training:
            model.train()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return enc_flops, total_flops


def _build_dataloader_worker_kwargs(num_workers: int, prefetch_factor: int) -> dict:
    if int(num_workers) <= 0:
        return {}
    kwargs = {"persistent_workers": True}
    if int(prefetch_factor) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _compute_mean_std_from_dataset(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **_build_dataloader_worker_kwargs(num_workers, prefetch_factor),
    )

    sum_per_channel = None
    sumsq_per_channel = None
    count = 0

    for batch in loader:
        samples = batch[0] if isinstance(batch, (list, tuple)) else batch
        if samples.ndim != 4:
            raise ValueError(f"Expected samples to be 4D (B,C,H,W), got shape={tuple(samples.shape)}")
        samples64 = samples.to(dtype=torch.float64)
        if sum_per_channel is None:
            channels = samples64.shape[1]
            sum_per_channel = torch.zeros(channels, dtype=torch.float64)
            sumsq_per_channel = torch.zeros(channels, dtype=torch.float64)
        sum_per_channel += samples64.sum(dim=(0, 2, 3))
        sumsq_per_channel += (samples64 * samples64).sum(dim=(0, 2, 3))
        count += int(samples64.shape[0] * samples64.shape[2] * samples64.shape[3])

    if sum_per_channel is None or sumsq_per_channel is None or count <= 0:
        raise RuntimeError("Failed to compute mean/std: empty dataset or invalid counts.")

    mean = sum_per_channel / float(count)
    var = (sumsq_per_channel / float(count)) - (mean * mean)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32), count


def _load_standardize_stats(stats_path: Path):
    with np.load(stats_path, allow_pickle=False) as loaded:
        if "mean" not in loaded or "std" not in loaded:
            raise KeyError("stats file must include 'mean' and 'std'.")

        mean_arr = np.asarray(loaded["mean"], dtype=np.float32)
        std_arr = np.asarray(loaded["std"], dtype=np.float32)
        if mean_arr.size == 0 or std_arr.size == 0:
            raise ValueError("stats file has empty mean/std arrays.")
        if mean_arr.shape != std_arr.shape:
            raise ValueError(
                f"stats mean/std shape mismatch: mean={mean_arr.shape}, std={std_arr.shape}"
            )

        def _scalar(key, default=None):
            if key not in loaded:
                return default
            arr = np.asarray(loaded[key])
            if arr.size == 0:
                return default
            v = arr.reshape(-1)[0]
            return v.item() if hasattr(v, "item") else v

        meta = {
            "input_height": _scalar("input_height", None),
            "input_width": _scalar("input_width", None),
            "preprocess": _scalar("preprocess", None),
            "count": _scalar("count", None),
        }

    return mean_arr, std_arr, meta


def _load_train_loss_history_from_log(log_path: Path):
    if not log_path.exists():
        return [], []

    latest_by_epoch = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" not in record or "train_loss" not in record:
                continue
            latest_by_epoch[int(record["epoch"])] = float(record["train_loss"])

    if not latest_by_epoch:
        return [], []

    epochs = sorted(latest_by_epoch.keys())
    losses = [latest_by_epoch[epoch] for epoch in epochs]
    return epochs, losses


def _save_train_loss_curve_from_log(output_dir: str, current_epoch: int) -> None:
    output_dir_path = Path(output_dir)
    log_path = output_dir_path / "log.txt"
    epochs, losses = _load_train_loss_history_from_log(log_path)
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = np.asarray(epochs, dtype=np.int64) + 1
    y_vals = np.asarray(losses, dtype=np.float32)
    ax.plot(x_vals, y_vals, linewidth=2.0, color="#1f77b4")
    ax.set_title(f"Training Loss (up to epoch {current_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    stable_path = output_dir_path / "train_loss_until_epoch_0000.png"
    fig.savefig(stable_path, dpi=150)
    plt.close(fig)


def main(args):
    misc.init_distributed_mode(args)
    _apply_dataset_preset(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_file_list, val_file_list = build_train_val_split(args)

    arr_range, arr_azimuth = _load_sensor_bins_for_args(args)
    range_limits = (float(arr_range.min()), float(arr_range.max()))
    az_vals_deg = np.degrees(-arr_azimuth)
    azimuth_limits = (float(az_vals_deg.min()), float(az_vals_deg.max()))

    input_size = (
        (args.input_height, args.input_width)
        if args.input_height and args.input_width
        else (args.input_size, args.input_size)
    )
    patch_size = _parse_patch_size(args.patch_size)
    stripe_patch_width = _parse_patch_size(args.stripe_patch_width)
    stripe_patch_stride = _parse_patch_size(args.stripe_patch_stride)
    if patch_size is not None:
        p_h, p_w = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        padded_h = ((input_size[0] + p_h - 1) // p_h) * p_h
        padded_w = ((input_size[1] + p_w - 1) // p_w) * p_w
        if misc.is_main_process():
            stripe_h, stripe_w = stripe_patch_width
            stride_h, stride_w = stripe_patch_stride
            print(
                f"Using custom patch size: {p_h}x{p_w} "
                f"(decoder padding: {input_size[0]}x{input_size[1]} -> {padded_h}x{padded_w})"
            )
            print(
                f"Axial stripe tokenizer: width={stripe_h}x{stripe_w}, "
                f"stride={stride_h}x{stride_w}"
            )

    base_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0)),
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
        ]
    )

    cache_dir_path = Path(args.cache_dir) if args.cache_dir else None
    use_cache = bool(args.use_cache) and cache_dir_path is not None and cache_dir_path.exists()
    if args.use_cache and cache_dir_path is not None and not cache_dir_path.exists() and misc.is_main_process():
        print(f"[cache] WARNING: cache_dir '{cache_dir_path}' does not exist; falling back to raw preprocessing.")
    dataset_cache_dir = str(cache_dir_path) if use_cache else None
    # When cache is on, cached files already include base_transform output; raw pipeline must be skipped.
    stats_transform = None if use_cache else base_transform

    do_standardize = not args.no_standardize
    mean_train = None
    std_train = None
    if misc.is_main_process():
        print(
            f"[input] preprocess={args.preprocess}, "
            f"ra_norm={'on' if args.ra_map_norm_enable else 'off'} "
            f"(mean={float(args.ra_map_norm_mean):.6f}, std={float(args.ra_map_norm_std):.6f}), "
            f"pretrain_axis={args.pretrain_axis}, "
            f"standardize={'on' if do_standardize else 'off'}, "
            f"cache={'on (' + str(cache_dir_path) + ')' if use_cache else 'off'}"
        )
        print(
            f"[loss] distribution={'on' if args.enable_dist_loss else 'off'}, "
            f"axis_routing={args.dist_loss_axis_routing}, "
            f"lambda_multipath={args.lambda_multipath}, "
            f"lambda_sidelobs={args.lambda_sidelobs}"
        )
    if do_standardize:
        stats_path = (
            Path(args.standardize_stats_path)
            if args.standardize_stats_path
            else (Path(args.output_dir) / "train_standardize_stats.npz")
        )

        if misc.is_main_process():
            should_recompute = bool(args.standardize_recompute or (not stats_path.exists()))
            recompute_reason = None
            if stats_path.exists() and not should_recompute:
                try:
                    mean_arr, std_arr, meta = _load_standardize_stats(stats_path)
                    expected_h = int(input_size[0])
                    expected_w = int(input_size[1])
                    saved_h = meta["input_height"]
                    saved_w = meta["input_width"]
                    saved_preprocess = meta["preprocess"]

                    if saved_preprocess is None:
                        should_recompute = True
                        recompute_reason = "missing preprocess metadata"
                    elif str(saved_preprocess) != str(args.preprocess):
                        should_recompute = True
                        recompute_reason = (
                            f"preprocess mismatch (saved={saved_preprocess}, current={args.preprocess})"
                        )
                    elif saved_h is None or saved_w is None:
                        should_recompute = True
                        recompute_reason = "missing input size metadata"
                    elif int(saved_h) != expected_h or int(saved_w) != expected_w:
                        should_recompute = True
                        recompute_reason = (
                            f"input size mismatch (saved={int(saved_h)}x{int(saved_w)}, current={expected_h}x{expected_w})"
                        )
                    elif mean_arr.shape[0] != int(args.in_chans):
                        should_recompute = True
                        recompute_reason = (
                            f"channel mismatch in stats (saved={mean_arr.shape[0]}, current in_chans={args.in_chans})"
                        )

                    if not should_recompute:
                        mean_train = torch.from_numpy(mean_arr).to(dtype=torch.float32)
                        std_train = torch.from_numpy(std_arr).to(dtype=torch.float32)
                        print(f"[standardize] Loaded train mean/std from {stats_path}")
                    else:
                        print(f"[standardize] Existing stats incompatible: {recompute_reason}. Recomputing.")
                except Exception as e:
                    should_recompute = True
                    recompute_reason = f"failed to load stats ({type(e).__name__}: {e})"
                    print(f"[standardize] Existing stats unusable: {recompute_reason}. Recomputing.")

            if should_recompute:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                # Step 1 (Calculation): load only the train set to compute mean/std (do not use val/test data).
                dataset_stats = RadarNpyDataset(
                    args.data_path,
                    file_list=train_file_list,
                    transform=stats_transform,
                    range_limits=range_limits,
                    azimuth_limits=azimuth_limits,
                    preprocess=args.preprocess,
                    ra_map_norm_enable=args.ra_map_norm_enable,
                    ra_map_norm_mean=args.ra_map_norm_mean,
                    ra_map_norm_std=args.ra_map_norm_std,
                    pretrain_axis=args.pretrain_axis,
                    range_bins=len(arr_range),
                    azimuth_bins=len(arr_azimuth),
                    cache_dir=dataset_cache_dir,
                    return_target=False,
                )
                mean_train, std_train, count = _compute_mean_std_from_dataset(
                    dataset_stats,
                    batch_size=max(1, int(args.batch_size)),
                    num_workers=max(0, int(args.num_workers)),
                    pin_memory=bool(args.pin_mem),
                    prefetch_factor=max(1, int(args.prefetch_factor)),
                )
                std_train = torch.clamp(std_train, min=float(args.standardize_eps))
                np.savez(
                    stats_path,
                    mean=mean_train.numpy(),
                    std=std_train.numpy(),
                    count=np.asarray(count, dtype=np.int64),
                    input_height=np.asarray(input_size[0], dtype=np.int64),
                    input_width=np.asarray(input_size[1], dtype=np.int64),
                    preprocess=np.asarray(args.preprocess),
                )
                print(f"[standardize] Computed and saved train mean/std to {stats_path}")

        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()

        if mean_train is None or std_train is None:
            mean_arr, std_arr, _ = _load_standardize_stats(stats_path)
            mean_train = torch.from_numpy(mean_arr).to(dtype=torch.float32)
            std_train = torch.from_numpy(std_arr).to(dtype=torch.float32)
            std_train = torch.clamp(std_train, min=float(args.standardize_eps))

        if misc.is_main_process():
            m_list = [float(x) for x in mean_train.flatten().tolist()]
            s_list = [float(x) for x in std_train.flatten().tolist()]
            print(f"[standardize] mean_train={m_list}, std_train={s_list}")

    if do_standardize:
        normalizer = transforms.Normalize(
            mean=[float(x) for x in mean_train.flatten().tolist()],
            std=[float(x) for x in std_train.flatten().tolist()],
        )
        if use_cache:
            transform_train = normalizer
            transform_vis = normalizer
        else:
            transform_train = transforms.Compose([base_transform, normalizer])
            transform_vis = transforms.Compose([base_transform, normalizer])
    else:
        if use_cache:
            transform_train = None
            transform_vis = None
        else:
            transform_train = base_transform
            transform_vis = base_transform

    dataset_train = RadarNpyDataset(
        args.data_path,
        file_list=train_file_list,
        transform=transform_train,
        range_limits=range_limits,
        azimuth_limits=azimuth_limits,
        preprocess=args.preprocess,
        ra_map_norm_enable=args.ra_map_norm_enable,
        ra_map_norm_mean=args.ra_map_norm_mean,
        ra_map_norm_std=args.ra_map_norm_std,
        pretrain_axis=args.pretrain_axis,
        range_bins=len(arr_range),
        azimuth_bins=len(arr_azimuth),
        cache_dir=dataset_cache_dir,
        return_target=False,
    )
    vis_files = val_file_list if len(val_file_list) > 0 else train_file_list
    dataset_for_vis = RadarNpyDataset(
        args.data_path,
        file_list=vis_files,
        transform=transform_vis,
        range_limits=range_limits,
        azimuth_limits=azimuth_limits,
        preprocess=args.preprocess,
        ra_map_norm_enable=args.ra_map_norm_enable,
        ra_map_norm_mean=args.ra_map_norm_mean,
        ra_map_norm_std=args.ra_map_norm_std,
        pretrain_axis=args.pretrain_axis,
        range_bins=len(arr_range),
        azimuth_bins=len(arr_azimuth),
        cache_dir=dataset_cache_dir,
    )
    print(f"Loaded train frames: {len(dataset_train)} (val hold-out: {len(val_file_list)}) from {args.data_path}")
    print(
        f"Dataset coordinate system: H=azimuth ({azimuth_limits[0]:.1f}° to {azimuth_limits[1]:.1f}°), "
        f"W=range ({range_limits[0]:.1f}m to {range_limits[1]:.1f}m)"
    )

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    wandb = None
    wandb_run = None
    if args.use_wandb and misc.is_main_process():
        wandb = _import_wandb()
        if wandb is None:
            print("[warn] wandb is not installed (or is shadowed by a local 'wandb/' folder). Disabling wandb logging.")
            args.use_wandb = False
        elif args.wandb_mode == "disabled":
            args.use_wandb = False
        else:
            os.environ.setdefault("WANDB_MODE", args.wandb_mode)
            os.environ.setdefault("WANDB_DIR", str(args.wandb_dir))
            run_name = args.wandb_name or f"{args.model}-mask{args.mask_ratio}-bs{args.batch_size}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=run_name,
                dir=str(args.wandb_dir),
                mode=args.wandb_mode,
            )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        **_build_dataloader_worker_kwargs(args.num_workers, args.prefetch_factor),
    )

    model_kwargs = dict(
        img_size=input_size,
        in_chans=args.in_chans,
        norm_pix_loss=args.norm_pix_loss,
        recon_loss_type=args.recon_loss,
        loss_weighting=args.loss_weighting,
        masking_mode=args.masking_mode,
        arr_range=arr_range,
        arr_azimuth=arr_azimuth,
        pretrain_axis=args.pretrain_axis,
        tokenization_mode=args.tokenization_mode,
        pos_embed_mode=args.pos_embed_mode,
        conv_stem_mode=args.conv_stem_mode,
        stripe_patch_width=stripe_patch_width,
        stripe_patch_stride=stripe_patch_stride,
        use_input_norm=not args.no_input_norm,
        enable_dist_loss=getattr(args, "enable_dist_loss", False),
        lambda_multipath=getattr(args, "lambda_multipath", 0.1),
        lambda_sidelobs=getattr(args, "lambda_sidelobs", 0.1),
        dist_loss_axis_routing=getattr(args, "dist_loss_axis_routing", "physical"),
        patch_block_min_size=args.patch_block_min_size,
        patch_block_max_size=args.patch_block_max_size,
        patch_block_max_steps=args.patch_block_max_steps,
    )
    if patch_size is not None:
        model_kwargs["patch_size"] = patch_size
    model = models_radarmae.__dict__[args.model](**model_kwargs)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # ---- FLOPs / params report ----
    if misc.is_main_process():
        total_p, trainable_p = _count_params(model_without_ddp)
        print(f"[params] total={total_p/1e6:.2f}M, trainable={trainable_p/1e6:.2f}M")
        enc_flops, total_flops = _measure_flops(
            model_without_ddp, device, input_size, args.in_chans, args.mask_ratio
        )
        if enc_flops is not None:
            print(
                f"[flops] (batch=1, input={input_size[0]}x{input_size[1]}, "
                f"mask_ratio={args.mask_ratio}) "
                f"encoder(visible-only)={enc_flops/1e9:.3f} GFLOPs, "
                f"full_forward(enc+dec+loss)={total_flops/1e9:.3f} GFLOPs"
            )
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            flops_stats = {
                "params_total_M": round(total_p / 1e6, 4),
                "params_trainable_M": round(trainable_p / 1e6, 4),
                "input_height": int(input_size[0]),
                "input_width": int(input_size[1]),
                "in_chans": int(args.in_chans),
                "mask_ratio": float(args.mask_ratio),
                "model": str(args.model),
                "tokenization_mode": str(getattr(args, "tokenization_mode", "")),
                "patch_size": str(getattr(args, "patch_size", "")),
                "stripe_patch_width": str(getattr(args, "stripe_patch_width", "")),
                "stripe_patch_stride": str(getattr(args, "stripe_patch_stride", "")),
                "encoder_visible_GFLOPs": round(enc_flops / 1e9, 4) if enc_flops is not None else None,
                "full_forward_GFLOPs": round(total_flops / 1e9, 4) if total_flops is not None else None,
            }
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps({"flops_stats": flops_stats}) + "\n")

    if args.finetune:
        print(f"Loading pretrained weights from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = checkpoint.get("model", checkpoint)
        patch_embed_module = getattr(model_without_ddp, "patch_embed", None)
        if (
            "patch_embed.proj.weight" in checkpoint_model
            and patch_embed_module is not None
            and hasattr(patch_embed_module, "proj")
        ):
            conv_weight = checkpoint_model["patch_embed.proj.weight"]
            if conv_weight.shape[1] == 3 and conv_weight.shape[1] != args.in_chans:
                print("Converting 3-channel patch_embed weights to 1-channel by averaging.")
                conv_weight = conv_weight.mean(dim=1, keepdim=True)
                checkpoint_model["patch_embed.proj.weight"] = conv_weight
            target_weight = patch_embed_module.proj.weight
            if conv_weight.shape[2:] != target_weight.shape[2:]:
                print(
                    f"Resizing patch_embed.proj.weight from {conv_weight.shape[2:]} "
                    f"to {target_weight.shape[2:]}."
                )
                resized = F.interpolate(
                    conv_weight.reshape(conv_weight.shape[0] * conv_weight.shape[1], 1, *conv_weight.shape[2:]),
                    size=target_weight.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
                conv_weight = resized.reshape(
                    target_weight.shape[0],
                    target_weight.shape[1],
                    *target_weight.shape[2:],
                )
                checkpoint_model["patch_embed.proj.weight"] = conv_weight
        if hasattr(model_without_ddp, "pos_embed") and "pos_embed" in checkpoint_model:
            interpolate_pos_embed(model_without_ddp, checkpoint_model)
        msg = model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Compile must wrap *after* DDP (if any) so the compiled forward sees the
    # DDP-wrapped module. fullgraph=False allows Dynamo to graph-break around
    # data-dependent control flow (e.g. weighted_block_masking's topk safety
    # net), which is required for our model to compile cleanly.
    if getattr(args, "compile", False):
        try:
            torch._dynamo.config.cache_size_limit = 64
            model = torch.compile(model, mode="default", fullgraph=False)
            print("[compile] torch.compile enabled (mode=default, fullgraph=False).")
        except Exception as e:  # graceful fallback; log and continue uncompiled
            print(f"[compile] disabled — {type(e).__name__}: {e}")

    decay, no_decay = [], []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=str(args.amp_dtype).lower() == "fp16")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    _save_training_args(args)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    args.train_start_wall_time = start_time
    args.total_train_steps = max(1, len(data_loader_train) * max(1, args.epochs - args.start_epoch))
    min_loss = float("inf")
    for epoch in range(args.start_epoch, args.epochs):
        epoch_profile_enabled = bool(args.profile_runtime and misc.is_main_process() and epoch < int(args.profile_epochs))
        epoch_profile = {}
        epoch_start_time = time.perf_counter() if epoch_profile_enabled else None
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_start_time = time.perf_counter() if epoch_profile_enabled else None
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if epoch_profile_enabled:
            epoch_profile["train_one_epoch"] = time.perf_counter() - train_start_time
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

        if args.output_dir and misc.is_main_process():
            current_loss = train_stats["loss"]
            if current_loss < min_loss:
                print(f"New best model found! Loss improved from {min_loss:.4f} to {current_loss:.4f}. Saving model...")
                min_loss = current_loss
                save_best_start = time.perf_counter() if epoch_profile_enabled else None
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch="best",
                )
                if epoch_profile_enabled:
                    epoch_profile["save_best_checkpoint"] = time.perf_counter() - save_best_start

            if (epoch + 1) % args.ckpt_interval == 0:
                print(f"Saving model at epoch {epoch} (every {args.ckpt_interval} epochs).")
                save_periodic_start = time.perf_counter() if epoch_profile_enabled else None
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=f"epoch_{epoch:04d}",
                )
                if epoch_profile_enabled:
                    epoch_profile["save_periodic_checkpoint"] = time.perf_counter() - save_periodic_start

            if epoch + 1 == args.epochs:
                print(f"Saving last model at epoch {epoch}...")
                save_last_start = time.perf_counter() if epoch_profile_enabled else None
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch="last",
                )
                if epoch_profile_enabled:
                    epoch_profile["save_last_checkpoint"] = time.perf_counter() - save_last_start

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                flush_start = time.perf_counter() if epoch_profile_enabled else None
                log_writer.flush()
                if epoch_profile_enabled:
                    epoch_profile["tensorboard_flush"] = time.perf_counter() - flush_start
            write_log_start = time.perf_counter() if epoch_profile_enabled else None
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            if epoch_profile_enabled:
                epoch_profile["write_log_json"] = time.perf_counter() - write_log_start
            should_save_loss_curve = (
                int(args.loss_curve_interval) > 0
                and (((epoch + 1) % int(args.loss_curve_interval)) == 0 or (epoch + 1) == args.epochs)
            )
            if should_save_loss_curve:
                loss_curve_start = time.perf_counter() if epoch_profile_enabled else None
                _save_train_loss_curve_from_log(args.output_dir, epoch)
                if epoch_profile_enabled:
                    epoch_profile["save_loss_curve"] = time.perf_counter() - loss_curve_start
            if wandb_run is not None:
                wandb_log_start = time.perf_counter() if epoch_profile_enabled else None
                wandb.log(log_stats, step=epoch)
                if epoch_profile_enabled:
                    epoch_profile["wandb_log"] = time.perf_counter() - wandb_log_start

        if (epoch % 50 == 0 or epoch + 1 == args.epochs) and args.output_dir and misc.is_main_process():
            recon_start = time.perf_counter() if epoch_profile_enabled else None
            fig = save_reconstruction_plot(
                model=model_without_ddp,
                dataset=dataset_for_vis,
                device=device,
                output_path=args.output_dir,
                epoch=epoch,
                img_index=0,
                mask_ratio=args.mask_ratio,
                range_roi=None,
                azimuth_roi=None,
                range_limits=getattr(dataset_for_vis, "range_limits", range_limits),
                azimuth_limits=getattr(dataset_for_vis, "azimuth_limits", azimuth_limits),
            )
            if epoch_profile_enabled:
                epoch_profile["save_reconstruction_plot"] = time.perf_counter() - recon_start
            if wandb_run is not None:
                wandb_recon_start = time.perf_counter() if epoch_profile_enabled else None
                wandb.log({"reconstruction_plot": wandb.Image(fig)}, step=epoch)
                if epoch_profile_enabled:
                    epoch_profile["wandb_reconstruction_plot"] = time.perf_counter() - wandb_recon_start

        if epoch_profile_enabled:
            epoch_profile["epoch_total"] = time.perf_counter() - epoch_start_time
            profile_parts = " ".join(
                f"{key}={value * 1000.0:.1f}ms"
                for key, value in sorted(epoch_profile.items(), key=lambda item: item[1], reverse=True)
            )
            print(f"[PROFILE][epoch {epoch}][epoch] {profile_parts}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    args._parser_argument_keys = tuple(action.dest for action in parser._actions if action.dest != "help")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir) if args.output_dir else Path("./output_dir")
    args.output_dir = str(base_output_dir / timestamp)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
