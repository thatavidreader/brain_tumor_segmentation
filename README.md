# Automated Brain Tumor Segmentation in MRI Scans
### Using Attention U-Net and the MONAI Framework

An end-to-end 3D semantic segmentation pipeline for multi-parametric MRI (mpMRI) brain tumor scans, built on the BraTS challenge format, PyTorch, and MONAI.

---

## 1. Problem Overview

Gliomas are heterogeneous tumors visible with different contrast across MRI modalities. The BraTS dataset provides four co-registered 3D modalities per patient:

| Modality | Highlights |
|---|---|
| FLAIR | Edema / whole tumor extent |
| T1 | Anatomical structure |
| T1ce (contrast-enhanced) | Active/enhancing tumor, necrotic core |
| T2 | Edema |

Voxel-wise labels: `0` background, `1` necrotic/non-enhancing core, `2` peritumoral edema, `4` enhancing tumor. Clinically, these are grouped into three **overlapping** evaluation regions:

- **Whole Tumor (WT)** = labels `{1, 2, 4}`
- **Tumor Core (TC)** = labels `{1, 4}`
- **Enhancing Tumor (ET)** = label `{4}`

Because WT ⊇ TC ⊇ ET, the model is trained as a **multi-label** (not mutually exclusive) segmentation problem: 3 output channels, each with an independent sigmoid, rather than a single softmax over 4 classes.

## 2. Why Attention U-Net

A standard 3D U-Net propagates *all* encoder features to the decoder via skip connections, including large amounts of background/normal-tissue signal that dilutes the tiny tumor regions. **Attention U-Net** (Oktay et al., 2018) inserts **Attention Gates** on each skip connection: a small gating network learns a spatial attention map, conditioned on the coarser decoder features, that suppresses irrelevant background activations before they reach the decoder. This is well-suited to BraTS, where tumor voxels are often <1% of the volume.

## 3. Why DiceFocal Loss

- **Dice Loss** directly optimizes the overlap metric used for evaluation and is naturally robust to class imbalance at the region level.
- **Focal Loss** down-weights easy (correctly classified) voxels and focuses gradient signal on hard, misclassified voxels — important when background voxels vastly outnumber tumor voxels.
- Combined (`monai.losses.DiceFocalLoss`), they give stable convergence and sharper boundaries than either loss alone.

## 4. Project Structure

```
brain_tumor_segmentation/
├── data/                     # raw/, mock/, and splits.json live here (gitignored)
├── src/
│   ├── dataset.py            # MONAI dict-transforms, CacheDataset, train/val/test split
│   ├── model.py              # Attention U-Net builder
│   ├── loss.py               # DiceFocalLoss builder
│   ├── metrics.py            # Dice, mIoU, HD95 (monai.metrics)
│   ├── train.py              # AMP training loop + validation + checkpointing
│   ├── evaluate.py           # Test-set evaluation + metric report + figures
│   ├── generate_mock_data.py # Synthetic BraTS-like NIfTI generator
│   └── utils.py              # Config/seed helpers, NIfTI I/O, visualizations
├── inference.py               # CLI for single-patient inference
├── config.yaml                # All hyperparameters and paths
├── requirements.txt
└── README.md
```

## 5. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (12+ GB VRAM for real BraTS volumes at 128³ patches); the pipeline also runs on CPU for the small synthetic dataset.

## 6. Quickstart — Smoke Test with Synthetic Data

You don't need the ~50GB real BraTS dataset to verify the pipeline works end-to-end. Generate a small synthetic dataset first:

```bash
python src/generate_mock_data.py --config config.yaml
```

This writes N synthetic cases (default 6) to `data/mock/`, each with 4 modality volumes and a nested tumor label map. Point the config at it:

```yaml
# config.yaml
paths:
  data_dir: "data/mock"
```

Then run training (a handful of epochs is enough to confirm the loop, losses, and checkpointing work):

```bash
python -m src.train --config config.yaml
```

Evaluate the best checkpoint on the held-out test split:

```bash
python -m src.evaluate --config config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Run inference on a single case:

```bash
python inference.py --config config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --case-dir data/mock/MockCase_000 \
    --case-id MockCase_000
```

## 7. Using Real BraTS Data

This project ships wired up to the **BraTS 2020 Training set** (via the Kaggle mirror `awsaf49/brats20-dataset-training-validation`), which contains 369 labeled cases. `_find_volume_file` in `src/dataset.py` accepts either `.nii` or `.nii.gz`, so files can be used as downloaded without recompressing.

```bash
pip install kaggle
kaggle auth login   # or supply an API token, see Kaggle CLI docs
kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p data/raw
unzip data/raw/brats20-dataset-training-validation.zip -d data/raw/extracted
mv data/raw/extracted/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_* data/raw/
rm -rf data/raw/extracted data/raw/brats20-dataset-training-validation.zip
```

Resulting in:
```
data/raw/BraTS20_Training_001/BraTS20_Training_001_flair.nii
data/raw/BraTS20_Training_001/BraTS20_Training_001_t1.nii
data/raw/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii
data/raw/BraTS20_Training_001/BraTS20_Training_001_t2.nii
data/raw/BraTS20_Training_001/BraTS20_Training_001_seg.nii
...
```

`paths.data_dir` in `config.yaml` already points at `data/raw`. For BraTS 2021/2023 (registration required at the official challenge portal), the same folder convention applies — just use `.nii.gz` files directly.

Before training on the full set: adjust `train.batch_size`, `data.cache_rate`, and `data.patch_size` to fit your available VRAM/RAM — the 369-case dataset (~31GB uncompressed) with `cache_rate: 0.5` can use tens of GB of RAM. Lower `cache_rate` (e.g. `0.1`–`0.2`) on memory-constrained machines. Then run `python -m src.train --config config.yaml`.

## 8. Pipeline Details

**Preprocessing (`src/dataset.py`)**: `LoadImaged` → `EnsureChannelFirstd` → `Orientationd` (canonical RAS) → `Spacingd` (resample to isotropic 1mm) → `NormalizeIntensityd` (per-channel z-score over non-zero/brain voxels) → modality stacking → BraTS label remapping to WT/TC/ET → (train only) `RandCropByPosNegLabeld` for foreground-biased 128³ patches, `RandFlipd` on all 3 axes, `RandShiftIntensityd`.

**Memory efficiency**: `CacheDataset` caches a configurable fraction of preprocessed volumes in RAM; training patches are small foreground-biased crops (not full volumes); AMP (`torch.cuda.amp.autocast` + `GradScaler`) halves activation memory and speeds up matmuls; validation/inference use `monai.inferers.sliding_window_inference` so full-resolution volumes never need to fit in VRAM at once.

**Metrics (`src/metrics.py`)**: Dice, mean IoU, and 95th-percentile Hausdorff Distance are computed per-region (WT/TC/ET) via `monai.metrics.DiceMetric`, `MeanIoU`, and `HausdorffDistanceMetric`, then averaged for a single headline number per epoch.

**Visualization (`src/utils.py`)**: `plot_slice_comparison` saves a 3-panel figure (MRI slice | ground truth overlay | prediction overlay) at the axial slice with the largest tumor extent — ready to drop into a presentation.

## 9. Reproducibility

All hyperparameters, paths, and the random seed live in `config.yaml`. `src/utils.set_seed` seeds Python, NumPy, and PyTorch (CPU + CUDA). The train/val/test case-ID split is cached to `data/splits.json` on first run.

## 10. Citation

- Oktay, O. et al. "Attention U-Net: Learning Where to Look for the Pancreas." arXiv:1804.03999 (2018).
- Baid, U. et al. "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation." arXiv:2107.02314 (2021).
- Cardoso, M.J. et al. "MONAI: An open-source framework for deep learning in healthcare imaging." arXiv:2211.02701 (2022).
