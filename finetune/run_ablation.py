#!/usr/bin/env python3
"""Paired ablations against the 11-epoch finetuning baseline.

Both modes keep the seed set fixed (2023/2024/2025) and the five pretrained
checkpoints fixed; exactly one training knob changes per mode, so a difference
against the baseline at the same seed can be attributed to that knob.

  short  --max-epoch 3        11 epochs cut to 3 (VAL_PER_EPOCH_FULL follows).
                              Less time for the encoder to drift away from its
                              pretrained state; everything else identical.
  probe  --freeze-encoder     Pretrained encoder frozen (weights + BN stats),
                              only BEV projection + detection head train
                              (6.88M / 134.19M = 5.1%). Epochs stay at 11 so
                              freezing is the only changed variable.

Runs serially on a single GPU. Already-finished runs are skipped, so the sweep
can be restarted without redoing work.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
CFG = ROOT / "configs" / "cfg_RadarMAE.yml"
RESULT_ROOT = ROOT.parent / "outputs" / "finetune" / "ablation"

CKPT_ROOT = ROOT.parent
CHECKPOINTS = [
    ("Full", CKPT_ROOT / "RadarMAE.pth"),
    ("L_A", CKPT_ROOT / "checkpoint_LA.pth"),
    ("L_R", CKPT_ROOT / "checkpoint_LR.pth"),
    ("L1w", CKPT_ROOT / "checkpoint_L1w.pth"),
    ("swap", CKPT_ROOT / "checkpoint-axis_swap.pth"),
]

SEEDS = [2023, 2024, 2025]

MODE_ARGS = {
    "short": ["--max-epoch", "3"],
    "probe": ["--freeze-encoder"],
}


def parse_final_all_bev_ap03(log_path: Path) -> float:
    """Final test-split BEV AP@IoU0.3 for Condition=all (same metric as the baseline sweep)."""
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    matches = re.findall(
        r"Condition:\s*all\b.*?IoU:\s*\[[^\]]*0\.3[^\]]*\].*?"
        r"BEV:\s*\[([^\]]+)\]",
        text,
        flags=re.DOTALL,
    )
    if not matches:
        raise RuntimeError(f"final Condition=all BEV result not found in {log_path}")
    values = [float(value.strip()) for value in matches[-1].split(",")]
    if len(values) < 3:
        raise RuntimeError(f"unexpected BEV result in {log_path}: {matches[-1]}")
    return values[2]


def already_done(log_path: Path):
    if not log_path.is_file():
        return None
    try:
        return parse_final_all_bev_ap03(log_path)
    except Exception:
        return None


def run_one(mode: str, seed: int, name: str, checkpoint: Path, gpu: str) -> float:
    seed_dir = RESULT_ROOT / mode / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / f"{name}.console.log"

    done = already_done(log_path)
    if done is not None:
        print(f"[{mode} seed {seed}] {name}: already finished, AP@0.3 = {done:.6f} (skipped)", flush=True)
        return done

    command = [
        str(PYTHON),
        str(ROOT / "main_train.py"),
        "--cfg", str(CFG),
        "--pretrain-path", str(checkpoint),
        "--classes", "Sedan",
        "--degrade-axis", "none",
        "--seed", str(seed),
        "--exp-name", f"{mode}_s{seed}_{name}",
        *MODE_ARGS[mode],
    ]
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    started = datetime.now()
    print(f"[{mode} seed {seed}] {started:%H:%M:%S} starting {name}: {checkpoint.name}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"[cmd] {' '.join(command)}\n\n")
        completed = subprocess.run(
            command, cwd=ROOT, env=env,
            stdout=log_file, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode}/{name} seed {seed} failed with exit code {completed.returncode}; see {log_path}"
        )

    ap = parse_final_all_bev_ap03(log_path)
    elapsed = (datetime.now() - started).total_seconds() / 60.0
    print(f"[{mode} seed {seed}] {name} BEV AP@0.3 = {ap:.6f} ({elapsed:.0f} min)", flush=True)
    return ap


def write_summary(mode: str, rows):
    summary = RESULT_ROOT / mode / "summary.tsv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    header = ["seed"] + [name for name, _ in CHECKPOINTS]
    with summary.open("w", encoding="utf-8") as output:
        output.write("\t".join(header) + "\n")
        for seed, values in rows:
            output.write(
                "\t".join([str(seed)] + [
                    f"{values[name]:.6f}" if name in values else "" for name, _ in CHECKPOINTS
                ]) + "\n"
            )
    print(f"[{mode}] summary written: {summary}", flush=True)


def run_mode(mode: str, seeds, gpu: str) -> int:
    rows = []
    failures = 0
    for seed in seeds:
        values = {}
        for name, checkpoint in CHECKPOINTS:
            try:
                values[name] = run_one(mode, seed, name, checkpoint, gpu)
            except Exception as exc:  # keep the sweep going; a hole is better than a stop
                failures += 1
                print(f"[{mode} seed {seed}] {name} FAILED: {exc}", flush=True)
            write_summary(mode, rows + [(seed, values)])
        rows.append((seed, values))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["short", "probe"], choices=sorted(MODE_ARGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the runs (default: inherit)")
    args = parser.parse_args()

    missing = [str(path) for _, path in CHECKPOINTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    total_failures = 0
    for mode in args.modes:
        print(f"\n===== mode={mode} seeds={args.seeds} gpu={args.gpu} "
              f"({len(args.seeds) * len(CHECKPOINTS)} runs) =====", flush=True)
        total_failures += run_mode(mode, args.seeds, args.gpu)

    print(f"\n[ablation] done at {datetime.now():%Y-%m-%d %H:%M:%S}, failures={total_failures}", flush=True)
    return 1 if total_failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ablation] stopped by user", flush=True)
        raise SystemExit(130)
