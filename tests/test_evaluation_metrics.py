"""
test_evaluation_metrics.py - Tests for metric calculation and clustered bootstrap.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluate import compute_comprehensive_metrics, compute_bootstrap_confidence_intervals


def test_compute_comprehensive_metrics_synthetic():
    """Verify accuracy, macro-F1, and per-class precision/recall on known array."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    
    m = compute_comprehensive_metrics(y_true, y_pred)
    assert m['accuracy'] == 1.0
    assert m['macro_f1'] == 1.0
    assert m['high_risk_recall'] == 1.0
    assert m['high_risk_precision'] == 1.0
    assert m['high_risk_specificity'] == 1.0


def test_compute_bootstrap_confidence_intervals():
    """Verify bootstrap confidence intervals return valid interval bounds."""
    test_df = pd.DataFrame({
        'Well_Name': ['W1', 'W1', 'W2', 'W2', 'W3', 'W3'],
        'true': [0, 1, 2, 0, 1, 2],
        'pred': [0, 1, 2, 0, 0, 2]
    })
    
    ci_out = compute_bootstrap_confidence_intervals(
        test_df,
        y_true=test_df['true'].values,
        y_pred=test_df['pred'].values,
        n_boot=100,
        seed=42
    )
    
    assert 'naive_row' in ci_out
    assert 'clustered_well' in ci_out
    row_acc_ci = ci_out['naive_row']['accuracy_ci']
    assert 0.0 <= row_acc_ci[0] <= row_acc_ci[1] <= 1.0
    clust_acc_ci = ci_out['clustered_well']['accuracy_ci']
    assert 0.0 <= clust_acc_ci[0] <= clust_acc_ci[1] <= 1.0
