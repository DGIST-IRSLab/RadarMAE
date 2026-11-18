# --------------------------------------------------------
# RadarMAE reconstruction visualization.
# Companion utility for RadarMAE pretraining (built on facebookresearch/mae —
# https://github.com/facebookresearch/mae). Renders original / masked / decoder
# panels for axial-token MAE outputs on RA-map inputs.
# --------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def _clamp_roi(roi, limits):
    if roi is None:
        return limits
    if len(roi) != 2:
        raise ValueError(f"ROI must contain two values, got {roi}")
    lo, hi = sorted(roi)
    min_limit, max_limit = limits
    lo = max(min_limit, lo)
    hi = min(max_limit, hi)
    if hi <= lo:
        raise ValueError(f"ROI {roi} becomes invalid after clamping to {limits}")
    return (lo, hi)


def _denormalize_for_plot(x, dataset):
    arr = np.asarray(x, dtype=np.float32)
    if bool(getattr(dataset, "ra_map_norm_enable", False)):
        mean = float(getattr(dataset, "ra_map_norm_mean", 0.0))
        std = float(getattr(dataset, "ra_map_norm_std", 1.0))
        arr = arr * std + mean
    return arr.astype(np.float32, copy=False)


def _convert_preprocessed_to_db(x, preprocess):
    arr = np.asarray(x, dtype=np.float32)
    mode = "none" if preprocess is None else str(preprocess).lower()
    eps = 1.0e-12
    if mode == "db":
        return arr.astype(np.float32, copy=False)
    if mode == "log1p":
        intensity = np.maximum(np.expm1(arr), 0.0)
        return (20.0 * np.log10(intensity + eps)).astype(np.float32, copy=False)
    if mode == "none":
        intensity = np.abs(arr)
        return (20.0 * np.log10(intensity + eps)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported preprocess='{preprocess}' for dB visualization.")


def _to_db_for_plot(x, dataset):
    denorm = _denormalize_for_plot(x, dataset)
    return _convert_preprocessed_to_db(denorm, getattr(dataset, "preprocess", None))


def _has_axis_mask_tensors(model, attr_names):
    for attr_name in attr_names:
        tensor = getattr(model, attr_name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            return False
    return True


def get_loss_mask_image(model, original_np, patch_mask=None):
    pixel_mask_tensor = getattr(model, "last_pixel_mask", None)
    if isinstance(pixel_mask_tensor, torch.Tensor) and pixel_mask_tensor.ndim == 4:
        pixel_mask = pixel_mask_tensor[0, 0].detach().cpu().numpy()
    elif patch_mask is not None:
        patch_mask_1d = patch_mask[0].detach().cpu().numpy()
        grid_h, grid_w = model.patch_embed.grid_size
        patch = model.patch_embed.patch_size
        p_h, p_w = patch if isinstance(patch, tuple) else (patch, patch)
        patch_grid = patch_mask_1d.reshape(grid_h, grid_w)
        pixel_mask = np.repeat(np.repeat(patch_grid, p_h, axis=0), p_w, axis=1)
    else:
        raise ValueError("Unable to build loss-mask visualization without pixel or patch mask data.")

    pixel_mask = pixel_mask[: original_np.shape[0], : original_np.shape[1]]
    fully_masked = pixel_mask >= (1.0 - 1.0e-6)
    return np.ma.array(original_np, mask=fully_masked)


def get_axis_mask_panels(model, original_np):
    masking_mode = str(getattr(model, "masking_mode", "")).lower().strip()
    preferred_specs = (
        ("last_pixel_mask_range", "Range block mask" if masking_mode == "weighted_block_masking" else "Range axis"),
        ("last_pixel_mask_azimuth", "Azimuth block mask" if masking_mode == "weighted_block_masking" else "Azimuth axis"),
    )
    fallback_specs = (
        ("last_pixel_mask_h", "H axial tokens"),
        ("last_pixel_mask_v", "V axial tokens"),
    )
    hint_specs = (
        ("last_pixel_hint_range", "Range hints"),
        ("last_pixel_hint_azimuth", "Azimuth hints"),
    )

    if _has_axis_mask_tensors(model, [attr for attr, _ in preferred_specs]):
        active_specs = preferred_specs
    elif _has_axis_mask_tensors(model, [attr for attr, _ in fallback_specs]):
        active_specs = fallback_specs
    else:
        return []

    panels = []
    for attr_name, label in active_specs:
        mask_tensor = getattr(model, attr_name)
        mask_np = mask_tensor[0, 0].detach().cpu().numpy()
        mask_bool = mask_np > 0.5
        ratio = float(mask_bool.mean())

        # Debug: print masking info.
        print(f"Debug - {attr_name}: shape={mask_np.shape}, ratio={ratio:.1%}, "
              f"min={mask_np.min():.4f}, max={mask_np.max():.4f}")

        panels.append({
            "image": np.ma.array(original_np, mask=mask_bool),
            "title": f"Masked ({label})",
            "ratio": ratio,
        })

    # Hint plots are disabled (per user request).
    # if masking_mode == "weighted_block_masking" and _has_axis_mask_tensors(model, [attr for attr, _ in hint_specs]):
    #     for attr_name, label in hint_specs:
    #         hint_tensor = getattr(model, attr_name)
    #         hint_bool = hint_tensor[0, 0].detach().cpu().numpy() > 0.0
    #         panels.append({
    #             "image": np.ma.array(original_np, mask=~hint_bool),
    #             "title": label,
    #             "ratio": float(hint_bool.mean()),
    #         })
    return panels


def save_reconstruction_plot(
    model,
    dataset,
    device,
    output_path,
    epoch,
    img_index=0,
    mask_ratio=0.75,
    range_roi=None,
    azimuth_roi=None,
    range_limits=(0.0, 118.0),
    azimuth_limits=(-53.0, 53.0),
):
    """
    Visualize the reconstruction from the given model and dataset, and save it to a file.

    Expected dataset format:
        - RA map: (H, W) = (azimuth, range)
            - H (height): azimuth axis
            - W (width): range axis
    """
    model.eval()

    sample = dataset[img_index]
    if isinstance(sample, (tuple, list)):
        input_tensor = sample[0]
        target_tensor = sample[1] if len(sample) > 1 else None
    else:
        input_tensor = sample
        target_tensor = None

    if target_tensor is input_tensor:
        target_tensor = None

    img_tensor = input_tensor.unsqueeze(0).to(device)
    target_tensor_gpu = target_tensor.unsqueeze(0).to(device) if isinstance(target_tensor, torch.Tensor) else None

    with torch.no_grad():
        loss, pred_output, mask = model(
            img_tensor,
            target_tensor_gpu,
            mask_ratio=mask_ratio,
        )

        # Check whether pred_output is in patch form ([N, L, p*p*C]) or image form ([N, C, H, W]).
        if pred_output.dim() == 3:  # patch form → unpatchify
             reconstructed_pred_img = model.unpatchify(pred_output) # [N, 1, H, W]
        elif pred_output.dim() == 4:  # already image form → use as-is
             reconstructed_pred_img = pred_output
        else:
             raise ValueError("Unexpected shape for prediction output from model.")

        # Keep a binary patch mask for the debug stats below.
        target_tensor_for_stats = target_tensor_gpu if target_tensor_gpu is not None else img_tensor[:, :1]
        mask_bin = (mask > 0.5).to(dtype=target_tensor_for_stats.dtype)

    target_tensor_for_plot = target_tensor if isinstance(target_tensor, torch.Tensor) else input_tensor[:1]
    original_np = _to_db_for_plot(target_tensor_for_plot.squeeze().cpu().numpy(), dataset)

    reconstructed_full_np = _to_db_for_plot(
        reconstructed_pred_img.squeeze().cpu().numpy(),
        dataset,
    )
    
    masked_cmap = plt.cm.viridis.copy()
    masked_cmap.set_bad(color=(0.6, 0.6, 0.6, 1.0))

    # Visualize the full masking pattern, combining H-stripe and V-stripe.
    # (H = azimuth axis, V = range axis.)
    mask_visualization = np.zeros_like(original_np)
    try:
        masking_mode = str(getattr(model, "masking_mode", "")).lower().strip()

        if masking_mode == "weighted_block_masking":
            # weighted_block_masking: combine range and azimuth masks.
            mask_range = getattr(model, "last_pixel_mask_range", None)
            mask_azimuth = getattr(model, "last_pixel_mask_azimuth", None)

            if isinstance(mask_range, torch.Tensor) and isinstance(mask_azimuth, torch.Tensor):
                mask_r = mask_range[0, 0].detach().cpu().numpy()
                mask_a = mask_azimuth[0, 0].detach().cpu().numpy()
                mask_r = mask_r[:original_np.shape[0], :original_np.shape[1]]
                mask_a = mask_a[:original_np.shape[0], :original_np.shape[1]]
                # A pixel is masked if either map marks it as masked.
                mask_visualization = np.maximum(mask_r, mask_a)
        else:
            # Axial: H and V pixel masks.
            mask_h = getattr(model, "last_pixel_mask_h", None)
            mask_v = getattr(model, "last_pixel_mask_v", None)

            if isinstance(mask_h, torch.Tensor) and isinstance(mask_v, torch.Tensor):
                mask_h_np = mask_h[0, 0].detach().cpu().numpy()
                mask_v_np = mask_v[0, 0].detach().cpu().numpy()
                mask_h_np = mask_h_np[:original_np.shape[0], :original_np.shape[1]]
                mask_v_np = mask_v_np[:original_np.shape[0], :original_np.shape[1]]
                # A pixel is masked if either map marks it as masked (OR).
                mask_visualization = np.maximum(mask_h_np, mask_v_np)
            else:
                # Fallback: build it from the patch mask.
                patch_mask_1d = mask[0].detach().cpu().numpy()
                grid_h, grid_w = model.patch_embed.grid_size
                patch = model.patch_embed.patch_size
                p_h, p_w = patch if isinstance(patch, tuple) else (patch, patch)
                patch_grid = patch_mask_1d.reshape(grid_h, grid_w)
                pixel_mask = np.repeat(np.repeat(patch_grid, p_h, axis=0), p_w, axis=1)
                mask_visualization = pixel_mask[:original_np.shape[0], :original_np.shape[1]]
    except Exception as e:
        print(f"Debug - mask visualization error: {e}")
        # Fallback to all-zero mask
        pass

    # Simple diagnostics: error on visible vs loss-mask patches
    try:
        with torch.no_grad():
            pred_patches_for_stats = model.patchify_target(reconstructed_pred_img)
            tgt_patches_for_stats = model.patchify_target(target_tensor_for_stats)
            per_patch_mae = torch.abs(pred_patches_for_stats - tgt_patches_for_stats).mean(dim=-1)  # [1, L]
            mask_t = mask_bin.to(per_patch_mae.dtype)
            masked_mae = (per_patch_mae * mask_t).sum() / (mask_t.sum() + 1e-6)
            visible_mae = (per_patch_mae * (1.0 - mask_t)).sum() / ((1.0 - mask_t).sum() + 1e-6)
        print(f"Debug - per-patch MAE: masked={masked_mae.item():.6f}, visible={visible_mae.item():.6f}")
        print(
            "Debug - value range:"
            f" target=[{float(target_tensor.min()):.6g}, {float(target_tensor.max()):.6g}]"
            f" pred=[{float(reconstructed_pred_img.min()):.6g}, {float(reconstructed_pred_img.max()):.6g}]"
        )
    except Exception as e:
        print(f"Debug - stats skipped: {e}")

    # Debug: print shape info to ensure coordinate assumptions hold
    print(f"Debug - Raw data shapes: original={original_np.shape}, mask_viz={mask_visualization.shape}, reconstructed={reconstructed_full_np.shape}")
    print(f"Debug - Raw data ordering: (H, W) = (azimuth, range)")
    print(f"Debug - Raw original: H={original_np.shape[0]} (azimuth bins), W={original_np.shape[1]} (range bins)")
    print(f"Debug - After transpose for plot: (range, azimuth)")
    print(f"Debug - Plot x-axis: azimuth, y-axis: range")
    print(f"Debug - Masking ratio: {float(mask_visualization.mean()):.1%}")

    # Check if original and reconstructed have same shape
    if original_np.shape != reconstructed_full_np.shape:
        print(f"Debug - WARNING: shape mismatch! Original {original_np.shape} vs Reconstructed {reconstructed_full_np.shape}")
    # mask_bool_np = mask.squeeze().bool().cpu().numpy()
    # final_vis_np = original_np.copy()
    # p = model.patch_embed.patch_size[0]
    # h = w = original_np.shape[0] // p

    # for i in range(h):
    #     for j in range(w):
    #         patch_index = i * w + j
    #         if mask_bool_np[patch_index]:
    #             final_vis_np[i*p:(i+1)*p, j*p:(j+1)*p] = reconstructed_pred_np[i*p:(i+1)*p, j*p:(j+1)*p]

    vmin = np.min(original_np)
    vmax = np.max(original_np)

    # Second plot: render masked pixels as gray and keep visible pixels as the original.
    masked_with_hint = original_np.copy()
    masked_with_hint = np.ma.array(
        masked_with_hint,
        mask=mask_visualization > 0.5  # Mask out masked pixels (rendered gray).
    )

    axis_panels = get_axis_mask_panels(model, original_np)
    for panel in axis_panels:
        print(f"Debug - {panel['title']}: ratio={panel['ratio']:.4f}")

    total_panels = 3 + len(axis_panels)
    plt.figure(figsize=(5 * total_panels, 5))

    range_extent = _clamp_roi(range_roi, range_limits) if range_roi is not None else range_limits
    azimuth_extent = _clamp_roi(azimuth_roi, azimuth_limits) if azimuth_roi is not None else azimuth_limits

    # MAE-paper-style paste: visible positions keep the original values,
    # masked positions are filled with the decoder's prediction.
    # (Visible-token decoder outputs are not supervised by the MAE loss
    #  and drift, so they are not plotted as part of the reconstruction.)
    mask_bool = mask_visualization > 0.5
    reconstructed_pasted = original_np.copy()
    if reconstructed_full_np.shape == original_np.shape:
        reconstructed_pasted[mask_bool] = reconstructed_full_np[mask_bool]
    else:
        print(
            f"Debug - paste skipped due to shape mismatch: "
            f"orig={original_np.shape}, recon={reconstructed_full_np.shape}"
        )
        reconstructed_pasted = reconstructed_full_np

    # Transpose images: (azimuth, range) → (range, azimuth).
    # Plot coordinates: x-axis=azimuth, y-axis=range.
    original_np_t = original_np.T
    masked_with_hint_t = masked_with_hint.T
    reconstructed_pasted_t = reconstructed_pasted.T

    # extent format: [xmin, xmax, ymin, ymax] where x=azimuth, y=range
    extent = [azimuth_extent[0], azimuth_extent[1], range_extent[0], range_extent[1]]

    panels = [
        (original_np_t, "Original RA Map", "viridis"),
        (masked_with_hint_t, "All Masked Tokens\n(gray=masked, color=visible)", masked_cmap),
    ]
    # Axis panels: render masked pixels as gray.
    for panel in axis_panels:
        # Append the ratio info to panel["title"].
        title_with_ratio = f"{panel['title']}\n({panel['ratio']*100:.1f}% masked)"
        # Transpose the axis panels too.
        panel_img_t = panel["image"].T if isinstance(panel["image"], np.ndarray) else panel["image"]
        panels.append((panel_img_t, title_with_ratio, masked_cmap))
    panels.append((reconstructed_pasted_t, "Reconstructed by MAE\n(visible=orig, masked=pred)", "viridis"))

    for idx, (img, title, cmap) in enumerate(panels):
        plt.subplot(1, len(panels), idx + 1)

        # Use a unified dB color range for all images.
        im = plt.imshow(
            img,
            cmap=cmap,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        plt.xlabel("Azimuth (°)")
        plt.ylabel("Range (m)")
        plt.title(title)
        plt.colorbar(im, label="dB")

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"reconstruction_epoch_{epoch:04d}.png")

    fig = plt.gcf()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Reconstruction plot for epoch {epoch} saved to {save_path}")

    model.train()

    return fig
