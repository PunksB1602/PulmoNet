import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import CheXpertDataset, get_transforms
from models import get_model
from utils import seed_everything, compute_AUROC

DATA_DIR = "./data" 
BATCH_SIZE = 16     
NUM_WORKERS = 4     
LR = 1e-4

# Paths and config: `DATA_DIR` should contain the `CheXpert-v1.0-small` folder.
# On Windows set `NUM_WORKERS = 0` to avoid dataloader hangs.


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        loop.set_description(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    return epoch_loss

def validate(model, loader, criterion, device, label_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epoch_loss = running_loss / len(loader)
    auroc = compute_AUROC(all_labels, all_preds, label_names)

    return epoch_loss, auroc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['densenet121', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = f"outputs/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    csv_root = os.path.join(DATA_DIR, "CheXpert-v1.0-small")
    train_dataset = CheXpertDataset(
        csv_file=os.path.join(csv_root, "train.csv"),
        root_dir=DATA_DIR, # Dataset.py joins this with path in CSV
        transform=get_transforms('train')
    )

    val_dataset = CheXpertDataset(
        csv_file=os.path.join(csv_root, "valid.csv"),
        root_dir=DATA_DIR,
        transform=get_transforms('valid')
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    model = get_model(args.model, num_classes=5).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_auroc = 0.0

    print(f"Starting training for {args.model}...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc = validate(model, val_loader, criterion, device, train_dataset.labels)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model Saved! (AUROC: {best_auroc:.4f})")

    print(f"\n Training Complete. Best AUROC: {best_auroc:.4f}")

# Note: BCEWithLogitsLoss is preferred over Sigmoid+BCELoss for numeric stability.
# The best model is written to `outputs/<model>/best_model.pth` during training.

if __name__ == "__main__":
    main()