"""
MONAI dictionary-transform data pipeline for BraTS-style mpMRI volumes.

Expected directory layout (per case):
    data/raw/<case_id>/<case_id>_flair.nii.gz
    data/raw/<case_id>/<case_id>_t1.nii.gz
    data/raw/<case_id>/<case_id>_t1ce.nii.gz
    data/raw/<case_id>/<case_id>_t2.nii.gz
    data/raw/<case_id>/<case_id>_seg.nii.gz
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    Spacingd,
)


class ConvertBratsLabelsd(MapTransform):
    """
    Convert raw BraTS integer labels {0, 1, 2, 4} into 3 overlapping
    binary channels expected by a sigmoid multi-label output head:
        channel 0 -> Whole Tumor (WT): labels 1, 2, 4
        channel 1 -> Tumor Core (TC):  labels 1, 4
        channel 2 -> Enhancing Tumor (ET): label 4
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        import torch

        d = dict(data)
        for key in self.keys:
            seg = d[key]
            wt = (seg == 1) | (seg == 2) | (seg == 4)
            tc = (seg == 1) | (seg == 4)
            et = seg == 4
            d[key] = torch.cat([wt, tc, et], dim=0).float()
        return d


def _find_volume_file(case_dir: Path, case_id: str, suffix: str) -> str | None:
    """Locate a case's NIfTI file, accepting either .nii.gz or plain .nii."""
    for ext in (".nii.gz", ".nii"):
        candidate = case_dir / f"{case_id}_{suffix}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def discover_cases(data_dir: str, modalities: list[str]) -> list[dict[str, str]]:
    """Scan `data_dir` for case subfolders and build MONAI data dicts."""
    data_dir = Path(data_dir)
    cases = []
    for case_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        case_id = case_dir.name
        item = {mod: _find_volume_file(case_dir, case_id, mod) for mod in modalities}
        item["label"] = _find_volume_file(case_dir, case_id, "seg")
        item["case_id"] = case_id
        if all(item[mod] for mod in modalities) and item["label"]:
            cases.append(item)
    return cases


def split_cases(
    cases: list[dict[str, str]],
    split_ratios: tuple[float, float, float],
    seed: int = 42,
    split_file: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split cases into train/val/test, optionally caching the split to disk."""
    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    if split_file:
        Path(split_file).parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(
                {
                    "train": [c["case_id"] for c in train],
                    "val": [c["case_id"] for c in val],
                    "test": [c["case_id"] for c in test],
                },
                f,
                indent=2,
            )
    return train, val, test


def get_transforms(config: dict[str, Any], mode: str = "train") -> Compose:
    """Build MONAI dictionary-transform pipelines for train/val/test."""
    modalities = config["data"]["modalities"]
    all_keys = modalities + ["label"]
    patch_size = config["data"]["patch_size"]

    load_and_format = [
        LoadImaged(keys=all_keys, image_only=True),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes=config["data"]["orientation"]),
        Spacingd(
            keys=all_keys,
            pixdim=config["data"]["spacing"],
            mode=["bilinear"] * len(modalities) + ["nearest"],
        ),
        NormalizeIntensityd(keys=modalities, nonzero=True, channel_wise=True),
        ConcatModalitiesd(keys=modalities, name="image"),
        ConvertBratsLabelsd(keys=["label"]),
        EnsureTyped(keys=["image", "label"]),
    ]

    if mode == "train":
        aug = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=config["train"]["pos_neg_ratio"][0],
                neg=config["train"]["pos_neg_ratio"][1],
                num_samples=config["train"]["samples_per_case"],
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        return Compose(load_and_format + aug)

    # val / test: no random cropping, full volume + sliding-window at inference
    return Compose(load_and_format)


class ConcatModalitiesd(MapTransform):
    """Stack single-channel modality tensors into one multi-channel 'image' tensor."""

    def __init__(self, keys, name: str = "image"):
        super().__init__(keys)
        self.name = name

    def __call__(self, data):
        import torch

        d = dict(data)
        stacked = torch.cat([d[k] for k in self.keys], dim=0).float()
        d[self.name] = stacked
        for k in self.keys:
            del d[k]
        return d


def get_dataloaders(
    config: dict[str, Any], data_dir: str | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders backed by MONAI CacheDataset."""
    data_dir = data_dir or config["paths"]["data_dir"]
    modalities = config["data"]["modalities"]

    cases = discover_cases(data_dir, modalities)
    if not cases:
        raise FileNotFoundError(
            f"No valid cases found under '{data_dir}'. "
            "Run `python src/generate_mock_data.py` for a smoke-test dataset, "
            "or point config.yaml 'paths.data_dir' at real BraTS data."
        )

    train_files, val_files, test_files = split_cases(
        cases,
        tuple(config["data"]["train_val_test_split"]),
        seed=config["train"]["seed"],
        split_file=config["paths"]["split_file"],
    )

    train_ds = CacheDataset(
        data=train_files,
        transform=get_transforms(config, "train"),
        cache_rate=config["data"]["cache_rate"],
        num_workers=config["data"]["num_workers"],
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_transforms(config, "val"),
        cache_rate=config["data"]["cache_rate"],
        num_workers=config["data"]["num_workers"],
    )
    test_ds = Dataset(data=test_files, transform=get_transforms(config, "test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config["data"]["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
