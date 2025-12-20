import torch
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score

def seed_everything(seed=42):
    # Make experiments reproducible: fixes seeds for Python, NumPy and PyTorch.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_AUROC(y_true, y_pred, label_names):
    # Compute AUROC per label and return the mean.
    # If a label has no positive/negative examples, it's skipped and not counted.
    auroc_vals = []
    for i in range(len(label_names)):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auroc_vals.append(score)
        except ValueError:
            pass

    return np.mean(auroc_vals) if auroc_vals else 0.5