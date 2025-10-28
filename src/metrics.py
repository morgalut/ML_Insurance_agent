# src/metrics.py
import numpy as np
from sklearn.metrics import average_precision_score

def auc_pr(y_true, y_score):
    """Compute AUC-PR."""
    return float(average_precision_score(y_true, y_score))

def precision_at_k(y_true, y_score, k_ratio):
    """Compute precision@k (percentage threshold)."""
    n = max(1, int(np.ceil(k_ratio * len(y_true))))
    idx = np.argsort(-y_score)[:n]
    return float(np.mean(np.array(y_true)[idx]))
