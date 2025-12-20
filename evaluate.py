import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataset import CheXpertDataset, get_transforms
from models import get_model
import os

def evaluate_models(model_names=['densenet121', 'resnet50', 'efficientnet_b0']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pathologies evaluated (order must match model output ordering)
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    val_dataset = CheXpertDataset(
        csv_file="./data/CheXpert-v1.0-small/valid.csv",
        root_dir="./data",
        transform=get_transforms('valid')
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    results = []

    for name in model_names:
        print(f"Evaluating {name}...")
        model = get_model(name, num_classes=5)
        # Expects trained checkpoints at `outputs/<model>/best_model.pth`
        checkpoint_path = f"outputs/{name}/best_model.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found for {name}, skipping.")
            continue
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        model_scores = {"Model": name}
        mean_auroc = 0
        for i, label in enumerate(labels):
            score = roc_auc_score(all_labels[:, i], all_preds[:, i])
            model_scores[label] = round(score, 4)
            mean_auroc += score
        
        model_scores["Mean AUROC"] = round(mean_auroc / len(labels), 4)
        results.append(model_scores)

    # Save per-model AUROC table to CSV for later inspection.
    df_results = pd.DataFrame(results)
    df_results.to_csv("outputs/final_evaluation_results.csv", index=False)
    
    print("\n" + "="*30)
    print("Final Results")
    print("="*30)
    print(df_results.to_markdown(index=False))
    print("="*30)

if __name__ == "__main__":
    evaluate_models()
    