"""
Run inference on a single patient's mpMRI scan with a trained Attention U-Net.

Usage:
    python inference.py \
        --config config.yaml \
        --checkpoint outputs/checkpoints/best_model.pth \
        --case-dir data/mock/MockCase_000 \
        --case-id MockCase_000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Orientationd, Spacingd

from src.dataset import ConcatModalitiesd, _find_volume_file
from src.model import build_model
from src.utils import (
    ensure_dirs,
    get_device,
    load_config,
    multilabel_to_brats_seg,
    plot_slice_comparison,
    save_nifti,
)


def build_inference_transforms(config: dict) -> Compose:
    modalities = config["data"]["modalities"]
    return Compose(
        [
            LoadImaged(keys=modalities, image_only=True),
            EnsureChannelFirstd(keys=modalities),
            Orientationd(keys=modalities, axcodes=config["data"]["orientation"]),
            Spacingd(keys=modalities, pixdim=config["data"]["spacing"], mode="bilinear"),
            NormalizeIntensityd(keys=modalities, nonzero=True, channel_wise=True),
            ConcatModalitiesd(keys=modalities, name="image"),
        ]
    )


def run_inference(config: dict, checkpoint_path: str, case_dir: str, case_id: str, out_dir: str) -> Path:
    device = get_device()
    ensure_dirs(out_dir)
    modalities = config["data"]["modalities"]

    case_dir = Path(case_dir)
    data_item = {mod: _find_volume_file(case_dir, case_id, mod) for mod in modalities}
    for mod, path in data_item.items():
        if not path:
            raise FileNotFoundError(f"Missing modality file for '{mod}' in {case_dir}")

    transforms = build_inference_transforms(config)
    processed = transforms(data_item)
    image = processed["image"].unsqueeze(0).to(device)  # add batch dim -> (1, C, D, H, W)

    # original affine/header for saving the prediction back in patient space
    import nibabel as nib

    ref_nii = nib.load(data_item[modalities[0]])
    affine = ref_nii.affine

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    roi = config["inference"]["sliding_window_roi"]
    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs=image,
            roi_size=roi,
            sw_batch_size=config["inference"]["sw_batch_size"],
            predictor=model,
            overlap=config["inference"]["overlap"],
            mode=config["inference"]["mode"],
        )
        probs = torch.sigmoid(outputs)
        preds = (probs > config["inference"]["prediction_threshold"]).float()

    pred_seg = multilabel_to_brats_seg(preds[0].cpu().numpy())
    out_path = Path(out_dir) / f"{case_id}_prediction.nii.gz"
    save_nifti(pred_seg, affine, out_path)
    print(f"Saved prediction NIfTI -> {out_path}")

    # Optional visualization if a ground-truth segmentation is available
    gt_path = _find_volume_file(case_dir, case_id, "seg")
    if gt_path:
        gt_seg = nib.load(gt_path).get_fdata().astype("uint8")
        image_np = image[0, 0].cpu().numpy()
        fig_path = Path(out_dir) / f"{case_id}_comparison.png"
        plot_slice_comparison(image_np, gt_seg, pred_seg, out_path=fig_path, title=f"Inference: {case_id}")
        print(f"Saved GT-vs-prediction figure -> {fig_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patient scan.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--case-dir", type=str, required=True, help="Folder containing the case's modality files")
    parser.add_argument("--case-id", type=str, required=True, help="Case ID prefix, e.g. 'MockCase_000'")
    parser.add_argument("--out-dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg, args.checkpoint, args.case_dir, args.case_id, args.out_dir)
