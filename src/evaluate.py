"""
Test-set evaluation: loads a trained checkpoint, runs sliding-window
inference over the held-out test split, and reports per-class and mean
Dice / IoU / HD95, saving both a JSON summary and comparison figures.

Usage:
    python -m src.evaluate --config config.yaml --checkpoint outputs/checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.metrics import SegmentationMetrics
from src.model import build_model
from src.utils import (
    ensure_dirs,
    get_device,
    load_config,
    multilabel_to_brats_seg,
    plot_slice_comparison,
)


def evaluate(config: dict, checkpoint_path: str, num_visualizations: int = 4) -> dict:
    device = get_device()
    ensure_dirs(config["paths"]["eval_dir"], config["paths"]["vis_dir"])

    _, _, test_loader = get_dataloaders(config)

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    metrics = SegmentationMetrics(class_names=config["data"]["class_names"])
    roi = config["inference"]["sliding_window_roi"]
    threshold = config["inference"]["prediction_threshold"]
    per_case_results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_id = batch.get("case_id", [f"case_{i}"])[0]

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=roi,
                sw_batch_size=config["inference"]["sw_batch_size"],
                predictor=model,
                overlap=config["inference"]["overlap"],
                mode=config["inference"]["mode"],
            )
            preds = (torch.sigmoid(outputs) > threshold).float()

            case_metrics = SegmentationMetrics(class_names=config["data"]["class_names"])
            case_metrics.update(preds, labels)
            case_result = case_metrics.aggregate()
            case_result["case_id"] = case_id
            per_case_results.append(case_result)

            metrics.update(preds, labels)

            if i < num_visualizations:
                image_np = images[0, 0].cpu().numpy()  # first modality channel for background
                gt_seg = multilabel_to_brats_seg(labels[0].cpu().numpy())
                pred_seg = multilabel_to_brats_seg(preds[0].cpu().numpy())
                plot_slice_comparison(
                    image_np,
                    gt_seg,
                    pred_seg,
                    out_path=Path(config["paths"]["vis_dir"]) / f"eval_{case_id}.png",
                    title=f"Case: {case_id}",
                )

    overall = metrics.aggregate()
    summary = {"overall": overall, "per_case": per_case_results}

    out_json = Path(config["paths"]["eval_dir"]) / "test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Test Set Results ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nFull report saved to {out_json}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-visualizations", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.num_visualizations)
