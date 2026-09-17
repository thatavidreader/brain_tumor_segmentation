"""
Training loop for Attention U-Net on BraTS-style mpMRI, with:
  - Automatic Mixed Precision (AMP) for reduced VRAM usage
  - Periodic sliding-window validation
  - Cosine LR scheduling, gradient clipping, early stopping
  - Checkpointing of the best model by mean validation Dice

Usage:
    python -m src.train --config config.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.loss import build_loss
from src.metrics import SegmentationMetrics
from src.model import build_model
from src.utils import ensure_dirs, get_device, load_config, plot_training_curves, set_seed


def validate(model, val_loader, loss_fn, metrics, device, config) -> dict:
    model.eval()
    metrics.reset()
    val_losses = []
    roi = config["inference"]["sliding_window_roi"]

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(enabled=config["train"]["amp"]):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi,
                    sw_batch_size=config["inference"]["sw_batch_size"],
                    predictor=model,
                    overlap=config["inference"]["overlap"],
                    mode=config["inference"]["mode"],
                )
                loss = loss_fn(outputs, labels)
            val_losses.append(loss.item())

            preds = (torch.sigmoid(outputs) > config["inference"]["prediction_threshold"]).float()
            metrics.update(preds, labels)

    results = metrics.aggregate()
    results["val_loss"] = sum(val_losses) / max(len(val_losses), 1)
    return results


def train(config: dict) -> None:
    set_seed(config["train"]["seed"])
    device = get_device()
    ensure_dirs(config["paths"]["checkpoint_dir"], config["paths"]["log_dir"], config["paths"]["vis_dir"])

    print(f"Using device: {device}")
    train_loader, val_loader, _ = get_dataloaders(config)

    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    metrics = SegmentationMetrics(class_names=config["data"]["class_names"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    scaler = GradScaler(enabled=config["train"]["amp"])

    best_dice = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    start_epoch = 1

    resume_path = Path(config["paths"]["checkpoint_dir"]) / "last_checkpoint.pth"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_dice = ckpt["best_dice"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from checkpoint: epoch {start_epoch}, best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['train']['epochs']}")

        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config["train"]["amp"]):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_dice": best_dice,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            resume_path,
        )

        if epoch % config["train"]["val_interval"] == 0:
            val_results = validate(model, val_loader, loss_fn, metrics, device, config)
            history["val_loss"].append(val_results["val_loss"])
            history["val_dice"].append(val_results["mean_dice"])

            print(
                f"  [Val] loss={val_results['val_loss']:.4f} "
                f"mean_dice={val_results['mean_dice']:.4f} "
                f"mean_iou={val_results['mean_iou']:.4f} "
                f"mean_hd95={val_results['mean_hd95']:.4f}"
            )

            if val_results["mean_dice"] > best_dice:
                best_dice = val_results["mean_dice"]
                epochs_without_improvement = 0
                ckpt_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "best_dice": best_dice},
                    ckpt_path,
                )
                print(f"  New best model saved (mean_dice={best_dice:.4f}) -> {ckpt_path}")
            else:
                epochs_without_improvement += config["train"]["val_interval"]

            if epochs_without_improvement >= config["train"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improvement} epochs).")
                break

    final_ckpt = Path(config["paths"]["checkpoint_dir"]) / "last_model.pth"
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, final_ckpt)
    plot_training_curves(history, out_path=Path(config["paths"]["vis_dir"]) / "training_curves.png")
    print(f"Training complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention U-Net for brain tumor segmentation.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = time.time()
    train(cfg)
    print(f"Total training time: {(time.time() - start) / 60:.1f} min")
