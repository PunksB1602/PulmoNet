import pandas as pd
import matplotlib.pyplot as plt
import os

# Expects `outputs/<model>/training_log.csv` with columns: epoch, train_loss, val_loss, val_auroc.
# Produces `outputs/model_comparison_results.png` summarizing loss and mean AUROC.

def plot_comparison(models=['densenet121', 'resnet50', 'efficientnet_b0']):
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = {'densenet121': '#1f77b4', 'resnet50': '#ff7f0e', 'efficientnet_b0': '#2ca02c'}

    for model_name in models:
        log_path = f"outputs/{model_name}/training_log.csv"

        if not os.path.exists(log_path):
            print(f"No log found for {model_name}. skipping...")
            continue

        df = pd.read_csv(log_path)
        epochs = range(1, len(df) + 1)

        axes[0].plot(epochs, df['train_loss'], '--', color=colors[model_name], alpha=0.5)
        axes[0].plot(epochs, df['val_loss'], '-', color=colors[model_name], label=f'{model_name}')
        axes[0].set_title('Training vs Validation Loss', fontsize=14)
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('BCE Loss')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        axes[1].plot(epochs, df['val_auroc'], '-o', color=colors[model_name], label=f'{model_name}')
        axes[1].set_title('Mean AUROC (5 Pathologies)', fontsize=14)
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('AUROC')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/model_comparison_results.png', dpi=300)
    print("Comparison plot saved to outputs/model_comparison_results.png")
    plt.show()

if __name__ == "__main__":
    plot_comparison()