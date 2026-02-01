from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(models=("densenet121", "resnet50", "efficientnet_b0"), artifacts_dir="./artifacts"):
    artifacts_dir = Path(artifacts_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    any_pr = False

    for model_name in models:
        log_path = artifacts_dir / model_name / "training_log.csv"
        if not log_path.exists():
            print(f"[SKIP] No log found for {model_name}: {log_path}")
            continue

        df = pd.read_csv(log_path)
        epochs = df["epoch"].to_numpy() if "epoch" in df.columns else list(range(1, len(df) + 1))

        # ---- Loss curves ----
        if "train_loss" in df.columns and "val_loss" in df.columns:
            axes[0].plot(epochs, df["train_loss"], linestyle="--", alpha=0.7, label=f"{model_name} (train)")
            axes[0].plot(epochs, df["val_loss"], linestyle="-", alpha=0.9, label=f"{model_name} (val)")
        else:
            print(f"[WARN] Missing loss columns in {log_path}")

        # ---- AUROC curves ----
        auroc_col = "val_macro_auroc" if "val_macro_auroc" in df.columns else ("val_auroc" if "val_auroc" in df.columns else None)
        if auroc_col is not None:
            axes[1].plot(epochs, df[auroc_col], marker="o", label=model_name)
        else:
            print(f"[WARN] Missing AUROC column in {log_path}")

        # ---- PR-AUC curves ----
        pr_col = "val_macro_pr_auc" if "val_macro_pr_auc" in df.columns else ("val_pr_auc" if "val_pr_auc" in df.columns else None)
        if pr_col is not None:
            any_pr = True
            axes[2].plot(epochs, df[pr_col], marker="o", label=model_name)
        else:
            print(f"[WARN] Missing PR-AUC column in {log_path}")

    # ---- Styling ----
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Validation Macro AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    axes[2].set_title("Validation Macro PR-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()

    fig.tight_layout()

    out_path = artifacts_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved comparison plot to: {out_path}")
    if not any_pr:
        print("[NOTE] None of the logs contained PR-AUC columns, so the PR-AUC plot may be empty.")


if __name__ == "__main__":
    plot_comparison()
