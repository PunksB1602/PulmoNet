import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42, deterministic: bool = True, strict: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict:
        # May raise if a non-deterministic op is used
        torch.use_deterministic_algorithms(True)


def compute_auroc(y_true, y_score, label_names):
    """
    Computes AUROC per label and macro AUROC.
    y_score should be probabilities (after sigmoid) or any monotonic score.
    """
    per_label_auroc = {}

    for i, label in enumerate(label_names):
        try:
            score = roc_auc_score(y_true[:, i], y_score[:, i])
            per_label_auroc[label] = float(score)
        except ValueError:
            per_label_auroc[label] = None

    valid = [v for v in per_label_auroc.values() if v is not None and not np.isnan(v)]
    macro_auroc = float(np.mean(valid)) if valid else None

    return macro_auroc, per_label_auroc
