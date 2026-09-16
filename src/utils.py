"""
Shared utilities: config loading, reproducibility, NIfTI I/O, and
visualization (GT vs. prediction overlays) for presentation figures.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml


def load_config(config_path: str | os.PathLike = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(*dirs: str | os.PathLike) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_nifti(volume: np.ndarray, affine: np.ndarray, out_path: str | os.PathLike) -> None:
    """Save a numpy array (D, H, W) as a compressed NIfTI file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(out_path))


def multilabel_to_brats_seg(pred_channels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Collapse 3-channel sigmoid predictions (WT, TC, ET) back into a single
    BraTS-style label map {0, 1, 2, 4} for visualization / saving.
    Priority: ET > TC(non-ET) > WT(non-TC) > background.
    """
    wt, tc, et = (pred_channels[c] > threshold for c in range(3))
    seg = np.zeros_like(wt, dtype=np.uint8)
    seg[wt] = 2          # edema
    seg[tc] = 1          # necrotic core (part of TC not ET)
    seg[et] = 4           # enhancing tumor
    return seg


def brats_seg_to_multilabel(seg: np.ndarray) -> np.ndarray:
    """Expand a BraTS label map {0,1,2,4} into 3 binary channels [WT, TC, ET]."""
    wt = np.isin(seg, [1, 2, 4])
    tc = np.isin(seg, [1, 4])
    et = seg == 4
    return np.stack([wt, tc, et], axis=0).astype(np.float32)


def plot_slice_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    slice_idx: int | None = None,
    out_path: str | os.PathLike = "outputs/visualizations/comparison.png",
    title: str = "",
) -> None:
    """
    Save a side-by-side figure: MRI slice | GT overlay | Prediction overlay.
    `image` is a single-modality 3D volume (D, H, W); `ground_truth` and
    `prediction` are label maps (D, H, W) with values in {0, 1, 2, 4}.
    """
    if slice_idx is None:
        # pick the slice with the largest tumor extent in the ground truth
        tumor_per_slice = (ground_truth > 0).reshape(ground_truth.shape[0], -1).sum(axis=1)
        slice_idx = int(np.argmax(tumor_per_slice))

    img_slice = image[slice_idx]
    gt_slice = ground_truth[slice_idx]
    pred_slice = prediction[slice_idx]

    cmap = plt.cm.get_cmap("nipy_spectral", 5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("MRI Slice")

    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap=cmap, vmin=0, vmax=4, alpha=0.5)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap=cmap, vmin=0, vmax=4, alpha=0.5)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    out_path: str | os.PathLike = "outputs/visualizations/training_curves.png",
) -> None:
    """Plot loss and Dice metric curves logged during training."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss Curve")

    if "val_dice" in history:
        axes[1].plot(history["val_dice"], label="Mean Val Dice", color="green")
    axes[1].set_xlabel("Validation Step")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].set_title("Validation Dice")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
