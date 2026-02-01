
import argparse
import json
import os
import random
from pathlib import Path

import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import CheXpertDataset, get_transforms
from src.models import get_model
from src.utils import seed_everything, compute_auroc

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


# Reproducible DataLoader 
def seed_worker(worker_id: int):
    # Ensure deterministic seeding for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g if num_workers > 0 else None,
    )


# Metrics
def compute_pr_auc_macro(y_true: np.ndarray, y_prob: np.ndarray, label_names):
    # Macro PR-AUC across labels, skip single-class labels
    per_label = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            per_label[name] = None
            continue
        per_label[name] = float(average_precision_score(yt, yp))

    valid = [v for v in per_label.values() if v is not None]
    macro = float(np.mean(valid)) if valid else None
    return macro, per_label


# Train / Val
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool):
    model.train()
    running_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device, label_names, use_amp: bool):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        loop.set_postfix(vloss=f"{loss.item():.4f}")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    val_loss = running_loss / max(1, len(loader))
    macro_auc, per_label_auc = compute_auroc(all_labels, all_probs, label_names)
    macro_pr, per_label_pr = compute_pr_auc_macro(all_labels, all_probs, label_names)

    return val_loss, macro_auc, per_label_auc, macro_pr, per_label_pr


# Plotting 
def save_curves(history_df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(history_df["epoch"], history_df["val_macro_auroc"], label="val_macro_auroc")
    ax2.set_title("Val Macro AUROC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(history_df["epoch"], history_df["val_macro_pr_auc"], label="val_macro_pr_auc")
    ax3.set_title("Val Macro PR-AUC")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PR-AUC")
    ax3.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=200)
    plt.close(fig)


# Subset sampling
def _patient_id_from_path(p: str) -> str:
    def extract_patients(csv_path):
        df = pd.read_csv(csv_path)
        paths = df["Path"] if "Path" in df.columns else df.iloc[:, 0]
        return set(_patient_id_from_path(p) for p in paths)

    parts = p.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("patient"):
            return part
    return "unknown"


def make_subset_patient_level(dataset: CheXpertDataset, fraction: float, seed: int):
    if not (0.0 < fraction < 1.0):
        return dataset

    paths = dataset._paths_raw  # dataset already stores raw paths from csv
    patient_ids = np.array([_patient_id_from_path(p) for p in paths])

    unique_patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)

    n_patients = max(1, int(len(unique_patients) * fraction))
    chosen = set(rng.choice(unique_patients, size=n_patients, replace=False).tolist())

    idx = [i for i, pid in enumerate(patient_ids) if pid in chosen]
    return Subset(dataset, idx)


def make_subset_image_level(dataset: CheXpertDataset, fraction: float, seed: int):
    if not (0.0 < fraction < 1.0):
        return dataset

    rng = np.random.default_rng(seed)
    n = max(1, int(len(dataset) * fraction))
    idx = rng.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, idx.tolist())

def extract_patients(csv_path: Path):
    df = pd.read_csv(csv_path)
    path_col = "Path" if "Path" in df.columns else df.columns[0]
    paths = df[path_col].astype(str)

    patients = set()
    for p in paths:
        p = p.replace("\\", "/")
        for part in p.split("/"):
            if part.startswith("patient"):
                patients.add(part)
                break
    return patients

# Main 
def main():
    # Add system and environment info to config
    import platform
    import torch.backends.cudnn as cudnn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["densenet121", "resnet50", "efficientnet_b0"],
                        help="Model architecture to use.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of training data to use (e.g., 0.1 = 10% of patients or images). 1.0 = full dataset.")
    parser.add_argument("--subset_mode", type=str, default="patient",
                        choices=["patient", "image"],
                        help="Subset mode: 'patient' (recommended, samples patients and all their images) or 'image' (samples individual images).")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP) for faster training on CUDA.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root folder containing CheXpert-v1.0-small/ directory.")
    parser.add_argument("--out_dir", type=str, default="./artifacts", help="Directory to save model checkpoints, logs, and outputs.")
    parser.add_argument("--uncertain_policy", type=str, default="ones", choices=["ones", "zeros"],
                        help="How to map CheXpert uncertain labels (-1): 'ones' (-1→1, default), 'zeros' (-1→0).")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for the classifier head.")
    args = parser.parse_args()

    config_dict = vars(args).copy()
    config_dict["python_version"] = platform.python_version()
    config_dict["platform"] = platform.platform()
    config_dict["pytorch_version"] = torch.__version__
    config_dict["cuda_available"] = torch.cuda.is_available()
    config_dict["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    config_dict["cudnn_version"] = cudnn.version() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        config_dict["gpu_name"] = torch.cuda.get_device_name(0)
        config_dict["gpu_count"] = torch.cuda.device_count()
        config_dict["gpu_capability"] = torch.cuda.get_device_capability(0)

    # Reproducibility
    seed_everything(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and (device.type == "cuda")

    # Start time logging
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Device: {device} | Model: {args.model} | AMP: {use_amp}")

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config_dict["device"] = str(device)
    config_dict["use_amp"] = bool(use_amp)
    config_dict["start_time"] = start_time_str
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    dataset_root = Path(args.data_dir) / "CheXpert-v1.0-small"
    train_csv = dataset_root / "train.csv"
    valid_csv = dataset_root / "valid.csv"


    # Check for patient overlap (data leakage) between train and val
    train_patients = extract_patients(train_csv)
    val_patients = extract_patients(valid_csv)
    overlap = train_patients & val_patients
    if len(overlap) > 0:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap)} shared patients between train and val")

    train_ds_full = CheXpertDataset(
        csv_file=str(train_csv),
        root_dir=str(dataset_root),
        transform=get_transforms("train"),
        labels=LABELS,
        uncertain_policy=args.uncertain_policy,
    )

    # Subset
    if args.subset < 1.0:
        if args.subset_mode == "patient":
            train_ds = make_subset_patient_level(train_ds_full, args.subset, args.seed)
            print(f"Using patient-level subset: {args.subset:.2f} ({len(train_ds)} samples)")
        else:
            train_ds = make_subset_image_level(train_ds_full, args.subset, args.seed)
            print(f"Using image-level subset: {args.subset:.2f} ({len(train_ds)} samples)")
    else:
        train_ds = train_ds_full

    val_ds = CheXpertDataset(
        csv_file=str(valid_csv),
        root_dir=str(dataset_root),
        transform=get_transforms("valid"),
        labels=LABELS,
        uncertain_policy=args.uncertain_policy,
    )

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, seed=args.seed)

    model = get_model(args.model, num_classes=len(LABELS), pretrained=True, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp)
        val_loss, macro_auc, per_label_auc, macro_pr, per_label_pr = validate(
            model, val_loader, criterion, device, LABELS, use_amp=use_amp
        )

        macro_auc_print = float(macro_auc) if macro_auc is not None else float("nan")
        macro_pr_print = float(macro_pr) if macro_pr is not None else float("nan")

        print(
            f"Train loss={train_loss:.4f} | Val loss={val_loss:.4f} | "
            f"Val AUROC={macro_auc_print:.4f} | Val PR-AUC={macro_pr_print:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": macro_auc_print,
            "val_macro_pr_auc": macro_pr_print,
        }

        for k in LABELS:
            a = per_label_auc.get(k, None)
            p = per_label_pr.get(k, None)
            row[f"val_auc_{k}"] = float(a) if a is not None else np.nan
            row[f"val_pr_{k}"] = float(p) if p is not None else np.nan

        history.append(row)

        # Save best by macro AUROC
        if macro_auc is not None and float(macro_auc) > best_auc:
            best_auc = float(macro_auc)
            ckpt = {
                "epoch": epoch,
                "best_macro_auroc": best_auc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config_dict,
            }
            torch.save(ckpt, out_dir / "best_model.pth")
            print(f"Saved new best model (macro AUROC={best_auc:.4f})")

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "training_log.csv", index=False)
        save_curves(hist_df, out_dir)


    # End time and duration logging
    end_time = time.time()
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_sec = end_time - start_time

    config_dict["end_time"] = end_time_str
    config_dict["training_time_seconds"] = round(elapsed_sec, 2)
    config_dict["training_time_minutes"] = round(elapsed_sec / 60, 2)

    # Overwrite config with timing info
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nDone. Best Macro AUROC: {best_auc:.4f}")
    print(f"Saved to: {out_dir} (best_model.pth, training_log.csv, curves.png)")


if __name__ == "__main__":
    main()
