"""
evaluate.py - Metric computation, confusion matrices, and bootstrap uncertainty estimation.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Features:
- Fast vectorized metric computation for 3-class classification
- High-risk class specific metrics (Precision, Recall, Specificity)
- Naive row-level bootstrap confidence intervals
- Clustered well-level bootstrap confidence intervals
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


CLASS_NAMES = ['Low/Normal', 'Medium', 'High']
CLASS_MAP = {'Low/Normal': 0, 'Medium': 1, 'High': 2}
INV_MAP = {0: 'Low/Normal', 1: 'Medium', 2: 'High'}


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes overall and per-class metrics including Specificity.
    Accepts string categories or integer labels.
    """
    if len(y_true) == 0:
        return {}

    # Convert to standard integers if strings
    if isinstance(y_true[0], str):
        y_t = np.array([CLASS_MAP[s] for s in y_true])
        y_p = np.array([CLASS_MAP[s] for s in y_pred])
    else:
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)

    cm = np.bincount(3 * y_t + y_p, minlength=9).reshape(3, 3)
    total = len(y_t)
    acc = np.trace(cm) / total

    row_sums = cm.sum(axis=1)  # true class supports
    col_sums = cm.sum(axis=0)  # predicted counts
    diag = np.diag(cm)

    prec = np.divide(diag, col_sums, out=np.zeros(3, dtype=float), where=col_sums > 0)
    rec = np.divide(diag, row_sums, out=np.zeros(3, dtype=float), where=row_sums > 0)
    
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros(3, dtype=float), where=denom > 0)
    macro_f1 = float(np.mean(f1))

    # Compute per-class specificity: TN / (TN + FP)
    spec = np.zeros(3, dtype=float)
    for c in range(3):
        # Mask out class c
        mask_other = [i for i in range(3) if i != c]
        tn = cm[np.ix_(mask_other, mask_other)].sum()
        fp = cm[mask_other, c].sum()
        spec[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            'precision': float(prec[c]),
            'recall': float(rec[c]),
            'f1_score': float(f1[c]),
            'specificity': float(spec[c]),
            'support': int(row_sums[c])
        }

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'high_risk_recall': float(rec[2]),
        'high_risk_precision': float(prec[2]),
        'high_risk_specificity': float(spec[2]),
        'high_risk_support': int(row_sums[2]),
        'total_samples': int(total)
    }


def _fast_acc_f1(y_t: np.ndarray, y_p: np.ndarray) -> Tuple[float, float, float]:
    """Ultra-fast vectorized computation of accuracy, macro-F1, and high-risk recall."""
    cm = np.bincount(3 * y_t + y_p, minlength=9).reshape(3, 3)
    total = len(y_t)
    if total == 0:
        return 0.0, 0.0, 0.0
    acc = np.trace(cm) / total
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    diag = np.diag(cm)
    prec = np.divide(diag, col_sums, out=np.zeros(3, dtype=float), where=col_sums > 0)
    rec = np.divide(diag, row_sums, out=np.zeros(3, dtype=float), where=row_sums > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros(3, dtype=float), where=denom > 0)
    return float(acc), float(np.mean(f1)), float(rec[2])


def compute_bootstrap_confidence_intervals(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    well_col: str = 'Well_Name',
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes both Naive Row-Level Bootstrap CIs and Clustered Well-Level Bootstrap CIs.
    Demonstrates the impact of spatial clustering on uncertainty bounds.
    """
    rng = np.random.RandomState(seed)
    n_rows = len(test_df)

    if isinstance(y_true[0], str):
        y_t_int = np.array([CLASS_MAP[s] for s in y_true])
        y_p_int = np.array([CLASS_MAP[s] for s in y_pred])
    else:
        y_t_int = np.asarray(y_true, dtype=int)
        y_p_int = np.asarray(y_pred, dtype=int)

    # 1. Naive row-level bootstrap
    row_accs = np.zeros(n_boot)
    row_f1s = np.zeros(n_boot)
    row_rec_h = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n_rows, n_rows)
        acc, f1, rec_h = _fast_acc_f1(y_t_int[idx], y_p_int[idx])
        row_accs[b] = acc
        row_f1s[b] = f1
        row_rec_h[b] = rec_h

    # 2. Clustered Well-Level Bootstrap
    clust_accs = np.zeros(n_boot)
    clust_f1s = np.zeros(n_boot)
    clust_rec_h = np.zeros(n_boot)

    if well_col in test_df.columns:
        well_dict = {
            w: (sub['t'].values, sub['p'].values)
            for w, sub in test_df.assign(t=y_t_int, p=y_p_int).groupby(well_col)
        }
        unique_wells = np.array(list(well_dict.keys()))
        n_wells = len(unique_wells)

        for b in range(n_boot):
            sampled_w = rng.choice(unique_wells, size=n_wells, replace=True)
            b_true = np.concatenate([well_dict[w][0] for w in sampled_w])
            b_pred = np.concatenate([well_dict[w][1] for w in sampled_w])
            acc, f1, rec_h = _fast_acc_f1(b_true, b_pred)
            clust_accs[b] = acc
            clust_f1s[b] = f1
            clust_rec_h[b] = rec_h
    else:
        clust_accs = row_accs
        clust_f1s = row_f1s
        clust_rec_h = row_rec_h

    return {
        'naive_row': {
            'accuracy_ci': (float(np.percentile(row_accs, 2.5)), float(np.percentile(row_accs, 97.5))),
            'macro_f1_ci': (float(np.percentile(row_f1s, 2.5)), float(np.percentile(row_f1s, 97.5))),
            'high_risk_recall_ci': (float(np.percentile(row_rec_h, 2.5)), float(np.percentile(row_rec_h, 97.5)))
        },
        'clustered_well': {
            'accuracy_ci': (float(np.percentile(clust_accs, 2.5)), float(np.percentile(clust_accs, 97.5))),
            'macro_f1_ci': (float(np.percentile(clust_f1s, 2.5)), float(np.percentile(clust_f1s, 97.5))),
            'high_risk_recall_ci': (float(np.percentile(clust_rec_h, 2.5)), float(np.percentile(clust_rec_h, 97.5)))
        }
    }
