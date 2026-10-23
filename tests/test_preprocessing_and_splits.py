"""
test_preprocessing_and_splits.py - Tests for leakage prevention and split integrity.
"""

import pytest
import pandas as pd
from src.data_loader import load_bridgeton_raw
from src.preprocessing import (
    prepare_chronological_split,
    prepare_spatiotemporal_split,
    get_grouped_kfold_splits,
    FEATURE_COLS_3GAS,
    TARGET_COL
)


def test_feature_cols_contain_no_leakage():
    """Verify that model feature set contains NO temperature or target proxies."""
    forbidden = [
        'Temperature_F', 'Temperature_C', 'Temp', 'Risk_Category',
        'Residual_N2_percent', 'Ratio_CH4_CO2', 'Combustion_Air_Index',
        'Pyrolysis_Index', 'Well_Name', 'Date_Sampled'
    ]
    for feat in FEATURE_COLS_3GAS:
        assert feat not in forbidden, f"Forbidden feature '{feat}' found in feature set!"
    assert FEATURE_COLS_3GAS == ['CH4_percent', 'CO2_percent', 'O2_percent']


def test_chronological_split_integrity():
    """Verify chronological split boundaries, sample counts, and purge window."""
    df = load_bridgeton_raw()
    split = prepare_chronological_split(df)
    
    assert len(split['X_train']) == 11247
    assert len(split['X_test']) == 3611
    
    # Verify temporal non-overlap
    max_train_date = split['train_df']['Date_Sampled'].max()
    min_test_date = split['test_df']['Date_Sampled'].min()
    assert max_train_date < min_test_date
    assert max_train_date < pd.to_datetime('2015-07-01')
    assert min_test_date >= pd.to_datetime('2016-03-01')


def test_spatiotemporal_split_zero_overlap():
    """Verify that the spatio-temporal holdout has zero well overlap and zero date overlap."""
    df = load_bridgeton_raw()
    split = prepare_spatiotemporal_split(df, seed=42)
    
    train_wells = set(split['train_df']['Well_Name'].unique())
    test_wells = set(split['test_df']['Well_Name'].unique())
    overlap = train_wells.intersection(test_wells)
    
    assert len(overlap) == 0, f"Well overlap found in spatio-temporal split: {overlap}"
    assert len(test_wells) == 30
    assert len(train_wells) == 120
    
    # Date non-overlap
    max_train_date = split['train_df']['Date_Sampled'].max()
    min_test_date = split['test_df']['Date_Sampled'].min()
    assert max_train_date < min_test_date


def test_grouped_kfold_zero_well_overlap():
    """Verify that GroupKFold partitions wells with zero cross-fold leakage."""
    df = load_bridgeton_raw()
    folds = get_grouped_kfold_splits(df, n_splits=5)
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_wells = set(df.iloc[train_idx]['Well_Name'].unique())
        val_wells = set(df.iloc[val_idx]['Well_Name'].unique())
        overlap = train_wells.intersection(val_wells)
        assert len(overlap) == 0, f"Fold {fold_idx} has well overlap: {overlap}"
