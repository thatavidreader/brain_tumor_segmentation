"""
Hybrid Dice + Focal loss for extreme foreground/background imbalance
in brain tumor segmentation (tumor voxels are a tiny fraction of the volume).
"""
from __future__ import annotations

from typing import Any

from monai.losses import DiceFocalLoss


def build_loss(config: dict[str, Any]) -> DiceFocalLoss:
    l = config["loss"]
    return DiceFocalLoss(
        sigmoid=l["sigmoid"],           # multi-label (overlapping WT/TC/ET) -> sigmoid, not softmax
        lambda_dice=l["lambda_dice"],
        lambda_focal=l["lambda_focal"],
        gamma=l["focal_gamma"],
        smooth_nr=l["smooth_nr"],
        smooth_dr=l["smooth_dr"],
    )
