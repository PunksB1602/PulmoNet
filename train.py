import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import CheXpertDataset, get_transforms
from models import get_model
from utils import seed_everything, compute_AUROC

DATA_DIR = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0
LR = 1e-4

scaler = torch.cuda.amp.GradScaler()

# Mixed precision: `scaler` enables AMP training for speed and memory savings.
# Use `--subset` to train on a small fraction for quick iteration/debugging.

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")

    return running_loss / len(loader)

def validate(model, loader, criterion, device, label_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
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
    parser.add_argument('--subset', type=float, default=0.1, help="Fraction of data to use (0.1 = 10%)")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model}")

    output_dir = f"outputs/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    csv_root = os.path.join(DATA_DIR, "CheXpert-v1.0-small")
    full_train_dataset = CheXpertDataset(
        csv_file=os.path.join(csv_root, "train.csv"),
        root_dir=DATA_DIR,
        transform=get_transforms('train')
    )

    if args.subset < 1.0:
        num_samples = int(len(full_train_dataset) * args.subset)
        indices = np.random.choice(len(full_train_dataset), num_samples, replace=False)
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Using {args.subset*100}% subset: {num_samples} images")
    else:
        train_dataset = full_train_dataset

    val_dataset = CheXpertDataset(
        csv_file=os.path.join(csv_root, "valid.csv"),
        root_dir=DATA_DIR,
        transform=get_transforms('valid')
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = get_model(args.model, num_classes=5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_auroc = 0.0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc = validate(model, val_loader, criterion, device, full_train_dataset.labels)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auroc': val_auroc
        })

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Best Model Saved!")

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
    print(f"\n Training Complete. Log saved to {output_dir}/training_log.csv")

if __name__ == "__main__":
    main()