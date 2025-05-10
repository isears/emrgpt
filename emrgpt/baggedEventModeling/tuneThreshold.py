# Determine what prediction threshold maximizes next-hour AUPRC / AUROC
import sys
from emrgpt.model import EventBasedEmrGPT
from emrgpt.baggedEventModeling.data import EventSequenceDS
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np
from torch.utils.data import DataLoader

import warnings

# Suppress only UndefinedMetricWarning
warnings.filterwarnings("ignore", module="sklearn")


def find_best_threshold_youden(y_true, y_prob):
    """
    Calculates the threshold that maximizes Youden's Index (Sensitivity + Specificity - 1).

    Parameters:
    - y_true: array-like of shape (n_samples,) — Ground-truth binary labels (0 or 1)
    - y_prob: array-like of shape (n_samples,) — Predicted probabilities for the positive class

    Returns:
    - best_threshold: float — Threshold that maximizes Youden's Index
    - best_j: float — Maximum Youden's Index value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youdens_j = tpr - fpr
    best_idx = np.argmax(youdens_j)
    best_threshold = thresholds[best_idx]
    best_j = youdens_j[best_idx]
    return best_threshold


if __name__ == "__main__":
    model_path = "./cache/EventBasedEmrGPT.pt"

    model = EventBasedEmrGPT(
        vocab_size=5018, n_embd=64, dropout=0.2, block_size=24, n_layer=10, n_head=6
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to("cuda")

    ds = EventSequenceDS(block_size=24)
    dl = DataLoader(dataset=ds, batch_size=32, num_workers=5, collate_fn=ds.collate_fn)

    y_hat = list()
    y_true = list()

    for idx, (x, y) in tqdm(enumerate(dl)):
        x.set_device("cuda")

        _, probabilities = model.generate(seed=x, lookahead=1)
        y_hat.append(probabilities.detach().cpu().squeeze(1))
        y_true.append(y[:, -1, :].bool())

    y_hat = torch.cat(y_hat, dim=0)
    y_true = torch.cat(y_true, dim=0)

    assert y_hat.shape == y_true.shape

    print(f"Overall AUROC: {roc_auc_score(y_true, y_hat, average='macro')}")
    print(f"Overall AUPRC: {average_precision_score(y_true, y_hat, average='macro')}")

    print("Calculating thresholds...")
    thresholds = torch.tensor(
        [
            find_best_threshold_youden(y_true[:, idx], y_hat[:, idx])
            for idx in tqdm(range(0, y_true.shape[1]))
        ]
    )

    missing_thresholds = (
        (torch.isnan(thresholds) | torch.isinf(thresholds)).sum().item()
    )
    success_rate = (len(thresholds) - missing_thresholds) / len(thresholds)

    print(
        f"Calculated {len(thresholds) - missing_thresholds} thresholds ({success_rate * 100:.1f}%)"
    )
    print(thresholds)
    thresholds[thresholds.isinf()] = thresholds.min()
    torch.save(thresholds, "cache/thresholds.pt")
