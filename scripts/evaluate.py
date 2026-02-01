import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import CheXpertDataset, get_transforms
from models import get_model

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def safe_roc_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _load_state_dict_from_ckpt(ckpt_obj):
    # Load state_dict from checkpoint (supports raw or dict format)
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        return ckpt_obj["model_state"], ckpt_obj
    return ckpt_obj, None


@torch.no_grad()
def eval_one_model(model_name: str, ckpt_path: Path, val_loader, device, eval_uncertain_policy: str):
    model = get_model(model_name, num_classes=len(LABELS), pretrained=False).to(device)

    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict, ckpt_meta = _load_state_dict_from_ckpt(ckpt_obj)
    model.load_state_dict(state_dict)
    model.eval()

    # Warn if checkpoint config disagrees with evaluation settings
    if ckpt_meta is not None and "config" in ckpt_meta:
        train_pol = ckpt_meta["config"].get("uncertain_policy", None)
        if train_pol is not None and train_pol != eval_uncertain_policy:
            print(
                f"[WARN] {model_name}: checkpoint trained with uncertain_policy='{train_pol}' "
                f"but evaluating with '{eval_uncertain_policy}'. Consider matching them."
            )

    all_probs = []
    all_targets = []

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    row = {"Model": model_name}

    aucs, prs = [], []
    for i, label in enumerate(LABELS):
        auc = safe_roc_auc(y_true[:, i], y_prob[:, i])
        pr = safe_pr_auc(y_true[:, i], y_prob[:, i])

        row[f"AUROC_{label}"] = round(auc, 4) if auc is not None else np.nan
        row[f"PR_AUC_{label}"] = round(pr, 4) if pr is not None else np.nan

        if auc is not None:
            aucs.append(auc)
        if pr is not None:
            prs.append(pr)

    row["Macro_AUROC"] = round(float(np.mean(aucs)), 4) if aucs else np.nan
    row["Macro_PR_AUC"] = round(float(np.mean(prs)), 4) if prs else np.nan

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["densenet121", "resnet50", "efficientnet_b0"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--uncertain_policy", type=str, default="ones", choices=["ones", "zeros"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_dir) / "CheXpert-v1.0-small"
    valid_csv = data_root / "valid.csv"

    val_dataset = CheXpertDataset(
        csv_file=str(valid_csv),
        root_dir=str(args.data_dir),
        transform=get_transforms("valid"),
        labels=LABELS,
        uncertain_policy=args.uncertain_policy,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    artifacts_dir = Path(args.artifacts_dir)
    results = []

    for name in args.models:
        ckpt_path = artifacts_dir / name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"[EVAL] {name} -> {ckpt_path}")
        row = eval_one_model(name, ckpt_path, val_loader, device, args.uncertain_policy)
        results.append(row)

    if not results:
        print("No models evaluated (no checkpoints found).")
        return

    df = pd.DataFrame(results)

    out_csv = artifacts_dir / "final_evaluation_results.csv"
    out_md = artifacts_dir / "final_evaluation_results.md"

    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False))

    print("\n" + "=" * 40)
    print("Final Evaluation Results")
    print("=" * 40)
    print(df.to_markdown(index=False))
    print("=" * 40)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
