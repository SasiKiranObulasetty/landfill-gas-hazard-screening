"""
preprocessing.py - Feature extraction, split definitions, and input validation.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Enforces zero-leakage rules:
- Strictly excludes Temperature_F, Temperature_C, Risk_Category proxies,
  Residual_N2_percent, engineered ratios, well identifiers, and dates from model inputs.
- Only uses directly measured sensor gases: CH4_percent, CO2_percent, O2_percent.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

# Canonical 3-gas feature set verified by Phase 10 audit
FEATURE_COLS_3GAS = ['CH4_percent', 'CO2_percent', 'O2_percent']
TARGET_COL = 'Risk_Category'

CLASS_ORDER = ['Low/Normal', 'Medium', 'High']
CLASS_TO_INT = {'Low/Normal': 0, 'Medium': 1, 'High': 2}
INT_TO_CLASS = {0: 'Low/Normal', 1: 'Medium', 2: 'High'}

# Physical plausible sensor measurement ranges
SENSOR_RANGES = {
    'CH4_percent': (0.0, 100.0),
    'CO2_percent': (0.0, 100.0),
    'O2_percent': (0.0, 25.0)
}


def prepare_chronological_split(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates the audited chronological split with an 8-month purge window:
    - Train: 2013-06-01 to 2015-06-30 (11,247 rows)
    - Purge/Val: 2015-07-01 to 2016-02-29 (244-day buffer)
    - Test:  2016-03-01 to 2016-10-31 (3,611 rows)
    """
    df_clean = df.dropna(subset=FEATURE_COLS_3GAS + [TARGET_COL, 'Date_Sampled']).copy()
    
    train_mask = df_clean['Date_Sampled'] < '2015-07-01'
    val_mask = (df_clean['Date_Sampled'] >= '2015-07-01') & (df_clean['Date_Sampled'] < '2016-03-01')
    test_mask = df_clean['Date_Sampled'] >= '2016-03-01'
    
    X_train = df_clean.loc[train_mask, FEATURE_COLS_3GAS].copy()
    y_train = df_clean.loc[train_mask, TARGET_COL].copy()
    
    X_val = df_clean.loc[val_mask, FEATURE_COLS_3GAS].copy()
    y_val = df_clean.loc[val_mask, TARGET_COL].copy()
    
    X_test = df_clean.loc[test_mask, FEATURE_COLS_3GAS].copy()
    y_test = df_clean.loc[test_mask, TARGET_COL].copy()
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'train_df': df_clean[train_mask],
        'test_df': df_clean[test_mask],
        'val_df': df_clean[val_mask],
        'feature_cols': FEATURE_COLS_3GAS,
        'target_col': TARGET_COL
    }


def prepare_spatiotemporal_split(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """
    Creates the strictest out-of-sample benchmark:
    - 120 training wells before 2015-07-01
    - 30 physically unseen test wells on/after 2016-03-01
    - Zero well overlap AND zero date overlap.
    """
    df_clean = df.dropna(subset=FEATURE_COLS_3GAS + [TARGET_COL, 'Date_Sampled', 'Well_Name']).copy()
    
    all_wells_sorted = sorted(df_clean['Well_Name'].unique())
    rng = np.random.RandomState(seed)
    shuffled_wells = rng.permutation(all_wells_sorted)
    
    spatial_train_wells = set(shuffled_wells[:120])
    spatial_test_wells = set(shuffled_wells[120:])
    
    train_mask = (df_clean['Date_Sampled'] < '2015-07-01') & (df_clean['Well_Name'].isin(spatial_train_wells))
    test_mask = (df_clean['Date_Sampled'] >= '2016-03-01') & (df_clean['Well_Name'].isin(spatial_test_wells))
    
    return {
        'X_train': df_clean.loc[train_mask, FEATURE_COLS_3GAS],
        'y_train': df_clean.loc[train_mask, TARGET_COL],
        'X_test': df_clean.loc[test_mask, FEATURE_COLS_3GAS],
        'y_test': df_clean.loc[test_mask, TARGET_COL],
        'train_wells': list(spatial_train_wells),
        'test_wells': list(spatial_test_wells),
        'train_df': df_clean[train_mask],
        'test_df': df_clean[test_mask]
    }


def get_grouped_kfold_splits(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates GroupKFold cross-validation splits grouped strictly by Well_Name."""
    df_clean = df.dropna(subset=FEATURE_COLS_3GAS + [TARGET_COL, 'Well_Name']).copy()
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df_clean, groups=df_clean['Well_Name']))


def validate_single_sample(ch4: float, co2: float, o2: float) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates a single set of gas readings:
    Returns (is_valid, error_message, warning_message).
    """
    # Check numeric types
    try:
        ch4 = float(ch4)
        co2 = float(co2)
        o2 = float(o2)
    except (ValueError, TypeError):
        return False, "All gas concentrations must be valid numeric values.", None

    # Check for NaN / Inf
    if np.isnan([ch4, co2, o2]).any() or np.isinf([ch4, co2, o2]).any():
        return False, "Gas concentration cannot be NaN or Infinite.", None

    # Check non-negative
    if ch4 < 0.0 or co2 < 0.0 or o2 < 0.0:
        return False, f"Negative concentration detected (CH4={ch4}%, CO2={co2}%, O2={o2}%). Gas concentrations must be >= 0%.", None

    # Check realistic physical upper bounds
    if ch4 > 100.0:
        return False, f"CH4 concentration ({ch4}%) exceeds maximum physical limit of 100.0%.", None
    if co2 > 100.0:
        return False, f"CO2 concentration ({co2}%) exceeds maximum physical limit of 100.0%.", None
    if o2 > 25.0:
        return False, f"O2 concentration ({o2}%) exceeds maximum atmospheric air oxygen level (25.0%).", None

    # Check gas sum
    gas_sum = ch4 + co2 + o2
    if gas_sum > 102.0:
        return False, f"Total measured gas concentration ({gas_sum:.1f}%) exceeds 100% (allowing 2% sensor tolerance). Sum of CH4 + CO2 + O2 cannot exceed 100%.", None

    warning = None
    if gas_sum < 40.0:
        warning = f"Notice: Sum of CH4 + CO2 + O2 is only {gas_sum:.1f}%. Balance gas / nitrogen exceeds {100.0 - gas_sum:.1f}%, which is unusually high for landfill biogas."
    elif gas_sum > 100.0:
        warning = f"Notice: Sum of CH4 + CO2 + O2 is {gas_sum:.1f}% (slightly exceeds 100% due to sensor calibration margin)."

    return True, None, warning


def validate_batch_dataframe(df_input: pd.DataFrame) -> Tuple[bool, pd.DataFrame, List[str]]:
    """
    Validates a batch DataFrame uploaded for gas screening:
    Returns (is_valid, cleaned_df, error_log).
    """
    errors = []
    
    # Check column names
    col_mapping = {}
    lower_cols = {c.lower().strip(): c for c in df_input.columns}
    
    # Map CH4
    for cand in ['ch4_percent', 'ch4', 'methane', 'methane_percent', 'ch4 (%)']:
        if cand in lower_cols:
            col_mapping[lower_cols[cand]] = 'CH4_percent'
            break
            
    # Map CO2
    for cand in ['co2_percent', 'co2', 'carbon_dioxide', 'co2 (%)']:
        if cand in lower_cols:
            col_mapping[lower_cols[cand]] = 'CO2_percent'
            break
            
    # Map O2
    for cand in ['o2_percent', 'o2', 'oxygen', 'o2 (%)']:
        if cand in lower_cols:
            col_mapping[lower_cols[cand]] = 'O2_percent'
            break

    missing = [c for c in FEATURE_COLS_3GAS if c not in col_mapping.values()]
    if missing:
        errors.append(f"Missing required gas columns: {missing}. Expected columns for CH4, CO2, and O2 concentrations.")
        return False, df_input, errors

    df_renamed = df_input.rename(columns=col_mapping)[FEATURE_COLS_3GAS].copy()

    # Convert to numeric
    for c in FEATURE_COLS_3GAS:
        df_renamed[c] = pd.to_numeric(df_renamed[c], errors='coerce')

    nan_rows = df_renamed.isna().any(axis=1).sum()
    if nan_rows > 0:
        errors.append(f"Found {nan_rows} rows with non-numeric or missing gas concentrations.")

    neg_rows = ((df_renamed['CH4_percent'] < 0) | (df_renamed['CO2_percent'] < 0) | (df_renamed['O2_percent'] < 0)).sum()
    if neg_rows > 0:
        errors.append(f"Found {neg_rows} rows with negative gas concentrations.")

    over_o2 = (df_renamed['O2_percent'] > 25.0).sum()
    if over_o2 > 0:
        errors.append(f"Found {over_o2} rows where O2 exceeds atmospheric limit of 25.0%.")

    gas_sum = df_renamed['CH4_percent'] + df_renamed['CO2_percent'] + df_renamed['O2_percent']
    over_sum = (gas_sum > 102.0).sum()
    if over_sum > 0:
        errors.append(f"Found {over_sum} rows where the sum of CH4 + CO2 + O2 exceeds 102.0%.")

    is_valid = len(errors) == 0
    return is_valid, df_renamed, errors
