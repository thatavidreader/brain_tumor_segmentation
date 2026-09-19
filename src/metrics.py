"""
Evaluation metrics for BraTS-style multi-label segmentation:
Dice Similarity Coefficient, mean IoU, and 95th-percentile Hausdorff Distance,
each reported per class (WT, TC, ET) and averaged.
"""
from __future__ import annotations

from typing import Any

import torch
from monai.metrics import ConfusionMatrixMetric, DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.metrics.confusion_matrix import get_confusion_matrix


class SegmentationMetrics:
    """Accumulates per-batch metrics across an epoch/dataset, then reduces."""

    def __init__(self, class_names: list[str] | None = None):
        self.class_names = class_names or ["WT", "TC", "ET"]
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.iou_metric = MeanIoU(include_background=True, reduction="mean_batch")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95, reduction="mean_batch"
        )
        self.accuracy_metric = ConfusionMatrixMetric(
            include_background=True, metric_name="accuracy", reduction="mean_batch"
        )
        self.raw_confusion: list[torch.Tensor] = []  # each item: (B, C, 4) = [tp, fp, tn, fn]

    def reset(self) -> None:
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd95_metric.reset()
        self.accuracy_metric.reset()
        self.raw_confusion = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        pred, target: (B, C, D, H, W) binary tensors (already thresholded),
        C = number of classes (WT, TC, ET).
        """
        self.dice_metric(y_pred=pred, y=target)
        self.iou_metric(y_pred=pred, y=target)
        # HD95's edge-detection step uses cucim/cupy on GPU tensors, whose JIT
        # compiler is broken on some environments (e.g. Kaggle's CUDA/cupy
        # version mismatch) - compute on CPU instead, which uses plain scipy.
        self.hd95_metric(y_pred=pred.cpu(), y=target.cpu())
        self.accuracy_metric(y_pred=pred, y=target)
        self.raw_confusion.append(get_confusion_matrix(y_pred=pred, y=target, include_background=True))

    def aggregate(self) -> dict[str, Any]:
        dice_per_class = self.dice_metric.aggregate()
        iou_per_class = self.iou_metric.aggregate()
        hd95_per_class = self.hd95_metric.aggregate()
        accuracy = self.accuracy_metric.aggregate()
        # (N, C, 4) across all update() calls -> mean over cases -> (C, 4) = [tp, fp, tn, fn]
        confusion_per_class = torch.cat(self.raw_confusion, dim=0).mean(dim=0)
        tp, fp, tn, fn = confusion_per_class.unbind(dim=-1)

        results: dict[str, Any] = {}
        for i, name in enumerate(self.class_names):
            results[f"dice_{name}"] = float(dice_per_class[i])
            results[f"iou_{name}"] = float(iou_per_class[i])
            hd95_val = float(hd95_per_class[i])
            results[f"hd95_{name}"] = hd95_val if hd95_val == hd95_val else float("nan")  # keep NaN visible
            results[f"tp_{name}"] = float(tp[i])
            results[f"fp_{name}"] = float(fp[i])
            results[f"fn_{name}"] = float(fn[i])
            results[f"tn_{name}"] = float(tn[i])
            results[f"accuracy_{name}"] = float(accuracy[i])

        results["mean_dice"] = float(dice_per_class.mean())
        results["mean_iou"] = float(iou_per_class.mean())
        finite_hd95 = hd95_per_class[torch.isfinite(hd95_per_class)]
        results["mean_hd95"] = float(finite_hd95.mean()) if len(finite_hd95) else float("nan")
        results["mean_accuracy"] = float(accuracy.mean())
        return results
