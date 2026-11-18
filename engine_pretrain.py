# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# Adapted for RadarMAE from facebookresearch/mae
# (https://github.com/facebookresearch/mae).
# --------------------------------------------------------
import math
import sys
import time
import datetime
from contextlib import nullcontext
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _should_profile_iter(args, epoch: int, data_iter_step: int) -> bool:
    if args is None or not getattr(args, "profile_runtime", False):
        return False
    profile_steps = max(0, int(getattr(args, "profile_steps", 0)))
    profile_epochs = max(0, int(getattr(args, "profile_epochs", 0)))
    if profile_steps <= 0 or profile_epochs <= 0:
        return False
    return epoch < profile_epochs and data_iter_step < profile_steps


def _accumulate_stats(dst, src) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value)


def _format_eta(seconds: float) -> str:
    return str(datetime.timedelta(seconds=max(0, int(seconds))))


def _estimate_training_finish(args, epoch: int, data_iter_step: int, steps_per_epoch: int):
    train_start_wall_time = getattr(args, "train_start_wall_time", None)
    total_train_steps = int(getattr(args, "total_train_steps", 0) or 0)
    start_epoch = int(getattr(args, "start_epoch", 0))
    if train_start_wall_time is None or total_train_steps <= 0 or steps_per_epoch <= 0:
        return None, None

    completed_steps = max(0, epoch - start_epoch) * steps_per_epoch + data_iter_step + 1
    if completed_steps <= 0:
        return None, None

    elapsed = max(0.0, time.time() - train_start_wall_time)
    avg_step_time = elapsed / completed_steps
    remaining_seconds = max(0.0, (total_train_steps - completed_steps) * avg_step_time)
    finish_time = datetime.datetime.fromtimestamp(train_start_wall_time + avg_step_time * total_train_steps)
    return remaining_seconds, finish_time


def _format_loss_terms(total_loss: float, loss_breakdown: dict) -> str:
    loss_parts = [f"total_loss={total_loss:.6f}"]
    for key, value in loss_breakdown.items():
        loss_parts.append(f"{key}={float(value):.6f}")
    return " ".join(loss_parts)


def _split_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Received an empty batch from the dataloader.")
        samples = batch[0]
        targets = batch[1] if len(batch) > 1 else None
        return samples, targets
    return batch, None


def _reduce_loss_breakdown(loss_breakdown: dict) -> dict:
    if not loss_breakdown:
        return {}

    reduced = {}
    tensor_items = []
    for key, value in loss_breakdown.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            tensor_items.append((key, value.detach().reshape(()).to(dtype=torch.float32)))
        else:
            reduced[key] = float(value)

    if tensor_items:
        keys = [key for key, _ in tensor_items]
        values = torch.stack([value for _, value in tensor_items], dim=0)
        if misc.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(values)
            values /= misc.get_world_size()
        reduced.update({key: float(value) for key, value in zip(keys, values.cpu().tolist())})

    return reduced


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter
    profile_step_sums = {}
    profile_model_sums = {}
    profile_count = 0
    profile_model = _unwrap_model(model)
    iter_time = misc.SmoothedValue(fmt="{avg:.4f}")
    data_time = misc.SmoothedValue(fmt="{avg:.4f}")
    epoch_wall_start = time.time()
    end = time.time()

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    num_steps = max(1, len(data_loader))
    for data_iter_step, batch in enumerate(data_loader):
        samples, targets = _split_batch(batch)
        data_time.update(time.time() - end)
        profile_this_iter = _should_profile_iter(args, epoch, data_iter_step)
        if profile_this_iter:
            _sync_device(device)
            iter_start = time.perf_counter()
            h2d_start = time.perf_counter()
            if hasattr(profile_model, "set_runtime_profile"):
                profile_model.set_runtime_profile(True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True) if isinstance(targets, torch.Tensor) else None
        if profile_this_iter:
            _sync_device(device)
            h2d_time = time.perf_counter() - h2d_start
            forward_start = time.perf_counter()

        amp_dtype = torch.bfloat16 if str(getattr(args, "amp_dtype", "fp16")).lower() == "bf16" else torch.float16
        autocast_context = (
            torch.cuda.amp.autocast(dtype=amp_dtype)
            if device.type == "cuda" and torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_context:
            loss, _, _ = model(
                samples,
                targets,
                mask_ratio=args.mask_ratio,
            )
        if profile_this_iter:
            _sync_device(device)
            forward_time = time.perf_counter() - forward_start

        loss_breakdown = dict(getattr(profile_model, "last_loss_breakdown", {}) or {})
        loss_breakdown_reduce = _reduce_loss_breakdown(loss_breakdown)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        if profile_this_iter:
            backward_step_start = time.perf_counter()
        loss_scaler(loss, optimizer, clip_grad=getattr(args, "clip_grad", None), parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if profile_this_iter:
            _sync_device(device)
            backward_step_time = time.perf_counter() - backward_step_start
            zero_grad_time = 0.0
        if (data_iter_step + 1) % accum_iter == 0:
            if profile_this_iter:
                zero_grad_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if profile_this_iter:
                _sync_device(device)
                zero_grad_time = time.perf_counter() - zero_grad_start

        #torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if loss_breakdown_reduce:
            metric_logger.update(**loss_breakdown_reduce)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            for key, value in loss_breakdown_reduce.items():
                log_writer.add_scalar(f"train_{key}", value, epoch_1000x)

        iter_time.update(time.time() - end)
        remaining_seconds, finish_time = _estimate_training_finish(args, epoch, data_iter_step, num_steps)
        log_parts = [
            f"{header} Iter [{data_iter_step + 1}/{num_steps}]",
            _format_loss_terms(loss_value_reduce, loss_breakdown_reduce),
            f"lr={lr:.6e}",
            f"time={iter_time.avg:.4f}s",
            f"data={data_time.avg:.4f}s",
        ]
        if remaining_seconds is not None and finish_time is not None:
            log_parts.append(f"train_eta={_format_eta(remaining_seconds)}")
            log_parts.append(f"finish={finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            log_parts.append(f"max_mem={memory_mb:.0f}MB")
        print("  ".join(log_parts))

        if profile_this_iter:
            _sync_device(device)
            total_time = time.perf_counter() - iter_start
            step_stats = {
                "h2d": h2d_time,
                "forward": forward_time,
                "backward_step": backward_step_time,
                "zero_grad": zero_grad_time,
                "total": total_time,
            }
            model_stats = dict(getattr(profile_model, "last_profile", {}) or {})
            if hasattr(profile_model, "set_runtime_profile"):
                profile_model.set_runtime_profile(False)
            _accumulate_stats(profile_step_sums, step_stats)
            _accumulate_stats(profile_model_sums, model_stats)
            profile_count += 1
            step_parts = " ".join(f"{key}={value * 1000.0:.1f}ms" for key, value in step_stats.items())
            print(f"[PROFILE][epoch {epoch} iter {data_iter_step}] {step_parts} loss={loss_value:.6f}")
            if model_stats:
                model_parts = " ".join(
                    f"{key}={value * 1000.0:.1f}ms"
                    for key, value in sorted(model_stats.items(), key=lambda item: item[1], reverse=True)
                )
                print(f"[PROFILE][epoch {epoch} iter {data_iter_step}][model] {model_parts}")
        end = time.time()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    epoch_total_time = time.time() - epoch_wall_start
    print(f"{header} Total time: {_format_eta(epoch_total_time)} ({epoch_total_time / num_steps:.4f} s / it)")
    print("Averaged stats:", metric_logger)
    if profile_count > 0:
        avg_step_parts = " ".join(
            f"{key}={profile_step_sums[key] / profile_count * 1000.0:.1f}ms"
            for key in sorted(profile_step_sums.keys())
        )
        print(f"[PROFILE][epoch {epoch}][avg_step] {avg_step_parts}")
        if profile_model_sums:
            avg_model_parts = " ".join(
                f"{key}={profile_model_sums[key] / profile_count * 1000.0:.1f}ms"
                for key in sorted(profile_model_sums.keys(), key=lambda item: profile_model_sums[item], reverse=True)
            )
            print(f"[PROFILE][epoch {epoch}][avg_model] {avg_model_parts}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
