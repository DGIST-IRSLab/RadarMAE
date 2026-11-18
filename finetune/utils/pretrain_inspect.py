"""Shared module that reads a pretrain ckpt and auto-aligns the MAE block of a finetune cfg.

Used by:
- `pipeline_detection_v1_0.py`, called automatically after cfg loading (only PRETRAIN_PATH is required).
- the batch launcher in `run_finetune_all_ckpts.py`, when generating per-ckpt cfgs with the same logic.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

CACHE_INPUT_HEIGHT = 112
CACHE_INPUT_WIDTH = 258
CACHE_PRETRAIN_AXIS = "azimuth_range"


def _parse_size_str(s: Any):
    """'4x16' / '16' / '2,3' / int / list → int or [int, int]."""
    if isinstance(s, int):
        return s
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s] if len(s) == 2 else int(s[0])
    text = str(s).strip().lower()
    if "x" in text:
        parts = text.split("x")
    elif "," in text:
        parts = text.split(",")
    else:
        try:
            return int(text)
        except ValueError:
            return text
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    return int(parts[0])


def inspect_checkpoint(ckpt_path) -> dict:
    """Extract MAE-block override info for finetuning from a pretrain ckpt."""
    import torch

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    a = state.get("args", None) if isinstance(state, dict) else None
    sd = state.get("model", state) if isinstance(state, dict) else state

    has_stripe_embed = isinstance(sd, dict) and any("stripe_embed" in k for k in sd)
    has_patch_embed = isinstance(sd, dict) and (
        "patch_embed.proj.weight" in sd or "patch_proj.weight" in sd
    )
    has_conv_stem = isinstance(sd, dict) and any(k.startswith("conv_stem") for k in sd)

    if a is not None and hasattr(a, "tokenization_mode"):
        mode = str(a.tokenization_mode).lower().strip()
    elif has_stripe_embed:
        mode = "axial"
    elif has_patch_embed:
        mode = "patch"
    else:
        mode = "unknown"

    info: dict = {"mode": mode}

    if a is not None:
        info["patch_size"] = _parse_size_str(getattr(a, "patch_size", 16))
        if mode == "axial":
            info["stripe_patch_width"] = _parse_size_str(getattr(a, "stripe_patch_width", "2x3"))
            info["stripe_patch_stride"] = _parse_size_str(getattr(a, "stripe_patch_stride", "2x3"))
        info["conv_stem_mode"] = str(getattr(a, "conv_stem_mode", "global")).lower()
        info["pos_embed_mode"] = str(getattr(a, "pos_embed_mode", "pe")).upper()
        info["use_input_norm"] = not bool(getattr(a, "no_input_norm", False))
        info["in_chans"] = int(getattr(a, "in_chans", 1))
        info["input_height"] = int(getattr(a, "input_height", 112))
        info["input_width"] = int(getattr(a, "input_width", 258))
        info["pretrain_axis"] = str(getattr(a, "pretrain_axis", "azimuth_range")).lower()
        # For dataset matching (K-Radar vs K-Radar_20).
        if getattr(a, "train_list", None):
            info["train_list"] = str(a.train_list)
        if getattr(a, "val_list", None):
            info["val_list"] = str(a.val_list)
        if getattr(a, "ra_map_norm_mean", None) is not None:
            info["ra_map_norm_mean"] = float(a.ra_map_norm_mean)
        if getattr(a, "ra_map_norm_std", None) is not None:
            info["ra_map_norm_std"] = float(a.ra_map_norm_std)
    else:
        info["patch_size"] = 16
        info["stripe_patch_width"] = [2, 3]
        info["stripe_patch_stride"] = [2, 3]
        info["conv_stem_mode"] = "per_token" if has_conv_stem else "global"
        info["pos_embed_mode"] = "PE"
        info["use_input_norm"] = True
        info["in_chans"] = 1
        info["input_height"] = 112
        info["input_width"] = 258
        info["pretrain_axis"] = "azimuth_range"

    if mode == "patch" and has_patch_embed:
        w = sd.get("patch_proj.weight", sd.get("patch_embed.proj.weight"))
        actual = (int(w.shape[2]), int(w.shape[3]))
        info["patch_size"] = actual[0] if actual[0] == actual[1] else list(actual)

    info["tokenizer_yaml"] = "stripe" if mode == "axial" else "patch"
    info["cache_compatible"] = (
        info["input_height"] == CACHE_INPUT_HEIGHT
        and info["input_width"] == CACHE_INPUT_WIDTH
        and info["pretrain_axis"] == CACHE_PRETRAIN_AXIS
    )
    return info


# ───────────────────────── cfg auto-apply (in-place) ─────────────────────────

# (info_key, cfg_key) — `cfg` here is the cfg.MODEL.BACKBONE.MAE block.
_INFO_TO_CFG = [
    ("tokenizer_yaml", "TOKENIZER"),
    ("patch_size", "PATCH_SIZE"),
    ("conv_stem_mode", "CONV_STEM_MODE"),
    ("pos_embed_mode", "POS_EMBED_MODE"),
    ("use_input_norm", "USE_INPUT_NORM"),
    ("in_chans", "IN_CHANS"),
    ("input_height", "IMG_HEIGHT"),
    ("input_width", "IMG_WIDTH"),
    ("pretrain_axis", "PRETRAIN_AXIS"),
]


def apply_to_finetune_cfg(cfg, ckpt_path=None, verbose: bool = True) -> list[tuple[str, Any, Any]]:
    """In-place override of cfg.MODEL.BACKBONE.MAE to match the ckpt's training args.

    If ckpt_path is None, cfg.MODEL.BACKBONE.MAE.PRETRAIN_PATH is used.
    Returns [(cfg_key, old_value, new_value), ...] — a change log.
    """
    mae = cfg.MODEL.BACKBONE.MAE

    if ckpt_path is None:
        ckpt_path = getattr(mae, "PRETRAIN_PATH", None)

    if not ckpt_path or not str(ckpt_path).strip():
        if verbose:
            print("[pretrain_inspect] PRETRAIN_PATH is empty → skipping auto-override.")
        return []

    if not os.path.exists(ckpt_path):
        if verbose:
            print(f"[pretrain_inspect] PRETRAIN_PATH does not exist: {ckpt_path} → skipping auto-override.")
        return []

    try:
        info = inspect_checkpoint(ckpt_path)
    except Exception as e:
        if verbose:
            print(f"[pretrain_inspect] ckpt inspect failed ({type(e).__name__}: {e}) → skipping auto-override.")
        return []

    changes: list[tuple[str, Any, Any]] = []

    def _set(key, new_value):
        old = mae.get(key, None) if hasattr(mae, "get") else getattr(mae, key, None)
        # Normalize list/tuple types so comparison is type-agnostic.
        old_norm = list(old) if isinstance(old, (list, tuple)) else old
        new_norm = list(new_value) if isinstance(new_value, (list, tuple)) else new_value
        if old_norm != new_norm:
            mae[key] = new_value
            changes.append((key, old, new_value))

    for info_key, cfg_key in _INFO_TO_CFG:
        if info_key in info:
            _set(cfg_key, info[info_key])

    if info["mode"] == "axial":
        if "stripe_patch_width" in info:
            _set("STRIPE_PATCH_WIDTH", info["stripe_patch_width"])
        if "stripe_patch_stride" in info:
            _set("STRIPE_PATCH_STRIDE", info["stripe_patch_stride"])

    # If the ra_map cache does not match the ckpt input shape, invalidate cache_dir (fall back to raw).
    if not info["cache_compatible"]:
        disabled = _disable_incompatible_ra_map_cache(cfg)
        for path in disabled:
            changes.append((path, "<cache>", ""))

    # Auto-match dataset split / normalize (K-Radar vs K-Radar_20).
    # Determine finetune split entries from the ckpt args' train_list/val_list paths.
    if "train_list" in info and "val_list" in info:
        ds_changes = _apply_dataset_to_cfg(
            cfg,
            train_list=info["train_list"],
            val_list=info["val_list"],
            mean=info.get("ra_map_norm_mean"),
            std=info.get("ra_map_norm_std"),
        )
        changes.extend(ds_changes)

    if verbose:
        print(f"[pretrain_inspect] ckpt={ckpt_path}")
        print(f"[pretrain_inspect] mode={info['mode']} tokenizer={info['tokenizer_yaml']} "
              f"patch={info.get('patch_size')} stripe_w={info.get('stripe_patch_width','-')} "
              f"stripe_s={info.get('stripe_patch_stride','-')} "
              f"stem={info['conv_stem_mode']} pe={info['pos_embed_mode']} "
              f"input=({info['input_height']}x{info['input_width']}) "
              f"axis={info['pretrain_axis']} use_input_norm={info['use_input_norm']} "
              f"cache_compat={info['cache_compatible']}")
        if changes:
            print(f"[pretrain_inspect] cfg overrides ({len(changes)}):")
            for k, old, new in changes:
                print(f"  - {k}: {old!r} → {new!r}")
        else:
            print("[pretrain_inspect] cfg already matches the ckpt — no changes.")

    return changes


def _apply_dataset_to_cfg(cfg, train_list: str, val_list: str,
                          mean: float | None = None, std: float | None = None) -> list[tuple[str, Any, Any]]:
    """Auto-match the finetune cfg's split + normalize from the ckpt's train_list/val_list.

    Pretraining only has [train, val] split files, so this infers a 'test*' file in the
    same directory and fills it in. If val and test resolve to the same file, only the
    two-split form is used so that test samples are not overwritten by val.

    Examples:
      pretrain train_list=.../split/train.txt → splits=[train.txt, test.txt, val.txt]
      pretrain train_list=.../split/train_1-20.txt → splits=[train_1-20.txt, test_1-20.txt]
    """
    changes: list[tuple[str, Any, Any]] = []

    # Checkpoint paths are machine-specific: keep only the split file names and
    # resolve them in this repository's finetune/resources/split/.
    base_dir = Path(__file__).resolve().parents[1] / "resources" / "split"
    train_path = base_dir / Path(train_list).name
    val_path = base_dir / Path(val_list).name

    # Infer the test split filename by replacing the first 'train' in train_list with 'test'.
    test_name = train_path.name.replace("train", "test", 1)
    test_path = base_dir / test_name
    if not test_path.exists():
        # No matching test file → reuse val (e.g. K-Radar_20, which has no separate hold-out).
        test_path = val_path

    if test_path == val_path:
        new_split = [str(train_path), str(test_path)]
    else:
        new_split = [str(train_path), str(test_path), str(val_path)]

    # cfg.DATASET.path_data.split override
    try:
        path_data = cfg.DATASET.path_data
    except AttributeError:
        return changes

    old_split = list(path_data.get("split", []))
    if old_split != new_split:
        path_data["split"] = new_split
        changes.append(("DATASET.path_data.split", old_split, new_split))

    # Override ra_map.normalize.mean / std (cfg layout: DATASET.ra_map.normalize, not path_data).
    if mean is not None or std is not None:
        try:
            normalize = cfg.DATASET.ra_map.normalize
        except AttributeError:
            return changes
        if mean is not None:
            old_mean = normalize.get("mean", None)
            if old_mean is None or abs(float(old_mean) - mean) > 1e-9:
                normalize["mean"] = mean
                changes.append(("DATASET.ra_map.normalize.mean", old_mean, mean))
        if std is not None:
            old_std = normalize.get("std", None)
            if old_std is None or abs(float(old_std) - std) > 1e-9:
                normalize["std"] = std
                changes.append(("DATASET.ra_map.normalize.std", old_std, std))

    return changes


def _disable_incompatible_ra_map_cache(cfg) -> list[str]:
    """Walk the cfg tree, find every ra_map.cache_dir, and clear it (raw fallback).

    Returns the list of disabled paths (e.g. 'DATASET.path_data.ra_map.cache_dir').
    """
    disabled: list[str] = []

    def _walk(node, path):
        if not hasattr(node, "items"):
            return
        for k, v in list(node.items()):
            sub_path = f"{path}.{k}" if path else str(k)
            if k == "ra_map" and hasattr(v, "items") and "cache_dir" in v:
                if v["cache_dir"]:
                    v["cache_dir"] = ""
                    disabled.append(f"{sub_path}.cache_dir")
            else:
                _walk(v, sub_path)

    _walk(cfg, "")
    return disabled
