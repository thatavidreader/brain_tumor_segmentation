"""
Synthetic BraTS-like dataset generator.

Creates a handful of fake patients with 4 MRI modalities + a segmentation
label map, matching the BraTS directory/naming convention, so the full
dataset -> model -> training -> evaluation pipeline can be exercised
end-to-end without downloading the real (~50GB) BraTS dataset.

Usage:
    python src/generate_mock_data.py --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from src.utils import load_config
except ImportError:  # allow running as `python src/generate_mock_data.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.utils import load_config


def make_synthetic_case(shape: tuple[int, int, int], rng: np.random.Generator):
    """Generate 4 correlated modality volumes and a plausible nested tumor label."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")

    # Smooth "brain" background intensity field, roughly ellipsoid-shaped.
    center = np.array([d, h, w]) / 2
    dist = np.sqrt(
        ((zz - center[0]) / (d / 2)) ** 2
        + ((yy - center[1]) / (h / 2)) ** 2
        + ((xx - center[2]) / (w / 2)) ** 2
    )
    brain_mask = dist < 0.95
    base = (1.0 - dist).clip(0, 1) * brain_mask

    # Random tumor center inside the brain.
    tumor_center = center + rng.uniform(-0.25, 0.25, size=3) * np.array([d, h, w])
    tumor_dist = np.sqrt(
        ((zz - tumor_center[0]) / (d * 0.12)) ** 2
        + ((yy - tumor_center[1]) / (h * 0.12)) ** 2
        + ((xx - tumor_center[2]) / (w * 0.12)) ** 2
    )
    enhancing = (tumor_dist < 0.5) & brain_mask
    necrotic = (tumor_dist < 0.8) & ~enhancing & brain_mask
    edema = (tumor_dist < 1.4) & ~enhancing & ~necrotic & brain_mask

    seg = np.zeros(shape, dtype=np.uint8)
    seg[edema] = 2
    seg[necrotic] = 1
    seg[enhancing] = 4

    modalities = {}
    # Each modality gets a distinct contrast response to the tumor sub-regions
    # plus independent Gaussian noise, mimicking mpMRI contrast differences.
    contrast_profiles = {
        "flair": {"edema": 0.9, "necrotic": 0.3, "enhancing": 0.5, "noise": 0.05},
        "t1": {"edema": 0.2, "necrotic": 0.1, "enhancing": 0.3, "noise": 0.05},
        "t1ce": {"edema": 0.2, "necrotic": 0.15, "enhancing": 0.95, "noise": 0.05},
        "t2": {"edema": 0.85, "necrotic": 0.4, "enhancing": 0.4, "noise": 0.05},
    }
    for mod, prof in contrast_profiles.items():
        vol = base.copy() * 0.6
        vol[edema] += prof["edema"]
        vol[necrotic] += prof["necrotic"]
        vol[enhancing] += prof["enhancing"]
        vol += rng.normal(0, prof["noise"], size=shape)
        vol = np.clip(vol, 0, None).astype(np.float32)
        modalities[mod] = vol

    return modalities, seg


def generate_mock_dataset(config: dict) -> None:
    mock_cfg = config["mock_data"]
    out_dir = Path(config["paths"]["mock_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(mock_cfg["seed"])
    affine = np.eye(4)
    shape = tuple(mock_cfg["volume_shape"])

    for i in range(mock_cfg["num_cases"]):
        case_id = f"MockCase_{i:03d}"
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        modalities, seg = make_synthetic_case(shape, rng)
        for mod, vol in modalities.items():
            nib.save(nib.Nifti1Image(vol, affine), str(case_dir / f"{case_id}_{mod}.nii.gz"))
        nib.save(nib.Nifti1Image(seg, affine), str(case_dir / f"{case_id}_seg.nii.gz"))

        print(f"Generated {case_id} -> {case_dir}")

    print(f"\nDone. {mock_cfg['num_cases']} synthetic cases written to '{out_dir}'.")
    print("Point config.yaml 'paths.data_dir' at this folder to smoke-test the pipeline:")
    print(f'  data_dir: "{out_dir.as_posix()}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic BraTS-like NIfTI data.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_mock_dataset(cfg)
