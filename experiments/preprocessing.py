"""
preprocessing.py - Clean, Reproducible Preprocessing Pipeline for Bridgeton Landfill Monitoring Data
Preserves original source datasets completely intact.
"""

import zipfile
import xml.etree.ElementTree as ET
import io
from pathlib import Path
import pandas as pd
import numpy as np

def load_bridgeton_data(source_path=None):
    """
    Loads k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx directly
    without modifying original file.
    Supports either direct xlsx path or extracting from datasets.zip.
    """
    if source_path is None:
        source_path = Path(r"c:\Users\obula\OneDrive\Desktop\4th_year_project\Confirmed_Minimal_Datasets\primary\k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx")
    else:
        source_path = Path(source_path)
        
    print(f"[Preprocessing] Loading dataset from: {source_path}")
    
    # Check if direct file exists
    if source_path.exists() and source_path.suffix == '.xlsx':
        with open(source_path, 'rb') as f:
            xlsx_bytes = f.read()
    else:
        # Fallback to datasets.zip
        zip_path = Path(r"c:\Users\obula\OneDrive\Desktop\4th_year_project\Datasets\datasets.zip")
        with zipfile.ZipFile(zip_path, 'r') as z:
            xlsx_bytes = z.read("datasets/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx")

    # Parse OpenXML sheetData directly (fast and memory-efficient)
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes), 'r') as xz:
        sheet_xml = xz.read("xl/worksheets/sheet1.xml")

    root = ET.fromstring(sheet_xml)
    sheet_data = root.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData")

    rows_data = []
    for r in sheet_data.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
        row_vals = {}
        for c in r.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
            ref = c.get('r')
            col_letter = "".join([ch for ch in ref if ch.isalpha()])
            t = c.get('t')
            if t == 'inlineStr':
                t_elem = c.find(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                val = t_elem.text if t_elem is not None else ""
            else:
                v_elem = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = v_elem.text if v_elem is not None else ""
            row_vals[col_letter] = val
        rows_data.append(row_vals)

    header_row = rows_data[0]
    col_map = {
        'A': 'Well_Name', 'B': 'Date_Sampled', 'C': 'CH4_percent', 'D': 'CO2_percent',
        'E': 'O2_percent', 'F': 'Residual_N2_percent', 'G': 'Temperature_F', 'H': 'Temperature_C',
        'I': 'Ratio_CH4_CO2', 'J': 'Risk_Category'
    }

    records = [{colname: r.get(letter, '') for letter, colname in col_map.items()} for r in rows_data[1:]]
    df = pd.DataFrame(records)

    # Cast numeric columns
    num_cols = ['CH4_percent', 'CO2_percent', 'O2_percent', 'Residual_N2_percent', 'Temperature_F', 'Temperature_C', 'Ratio_CH4_CO2']
    for nc in num_cols:
        df[nc] = pd.to_numeric(df[nc], errors='coerce')

    df['Date_Sampled'] = pd.to_datetime(df['Date_Sampled'], errors='coerce')
    
    # Sort chronologically by Well_Name and Date_Sampled
    df = df.sort_values(by=['Date_Sampled', 'Well_Name']).reset_index(drop=True)
    
    print(f"[Preprocessing] Successfully loaded {len(df)} rows across {df['Well_Name'].nunique()} wells.")
    print(f"[Preprocessing] Date range: {df['Date_Sampled'].min()} to {df['Date_Sampled'].max()}")
    
    return df

def build_experiment_A_data(df):
    """
    Experiment A: Gas-Only Contemporaneous Hazard Classification.
    Excludes Temperature_F and Temperature_C to prevent deterministic target leakage.
    """
    df_expA = df.copy()
    
    # Engineered gas ratio interaction features
    df_expA['Combustion_Air_Index'] = df_expA['Residual_N2_percent'] / (df_expA['O2_percent'] + 0.1)
    df_expA['Pyrolysis_Index'] = (df_expA['Residual_N2_percent'] / 20.0) / (df_expA['Ratio_CH4_CO2'] + 0.1)
    
    feature_cols = [
        'CH4_percent', 'CO2_percent', 'O2_percent', 'Residual_N2_percent',
        'Ratio_CH4_CO2', 'Combustion_Air_Index', 'Pyrolysis_Index'
    ]
    target_col = 'Risk_Category'
    
    # Chronological Split:
    # Train: 2013-06-01 to 2015-06-30 (~58.9%)
    # Val:   2015-07-01 to 2016-02-29 (~20.3%)
    # Test:  2016-03-01 to 2016-10-31 (~20.8%)
    train_mask = df_expA['Date_Sampled'] < '2015-07-01'
    val_mask = (df_expA['Date_Sampled'] >= '2015-07-01') & (df_expA['Date_Sampled'] < '2016-03-01')
    test_mask = df_expA['Date_Sampled'] >= '2016-03-01'
    
    X_train = df_expA.loc[train_mask, feature_cols]
    y_train = df_expA.loc[train_mask, target_col]
    
    X_val = df_expA.loc[val_mask, feature_cols]
    y_val = df_expA.loc[val_mask, target_col]
    
    X_test = df_expA.loc[test_mask, feature_cols]
    y_test = df_expA.loc[test_mask, target_col]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'full_df': df_expA
    }

def build_experiment_B_data(df):
    """
    Experiment B: Seven-Day-Ahead Risk Classification.
    Uses strictly past historical features (including temperature at time t and rolling rates of change).
    Target is Risk_Category at t + 7 days (tolerance: 5 to 9 days).
    Strictly avoids future leakage.
    """
    df_sorted = df.sort_values(by=['Well_Name', 'Date_Sampled']).reset_index(drop=True)
    
    # 1. Compute backward-looking features strictly within each well's history
    engineered_groups = []
    for well, group in df_sorted.groupby('Well_Name'):
        g = group.sort_values('Date_Sampled').copy()
        
        # Inter-sample time delta in days
        time_delta_days = g['Date_Sampled'].diff().dt.total_seconds() / (24 * 3600)
        time_delta_days = time_delta_days.replace(0, np.nan)
        
        # Historical rates of change strictly looking backward
        g['delta_temp_f'] = g['Temperature_F'].diff() / time_delta_days
        g['delta_ratio'] = g['Ratio_CH4_CO2'].diff() / time_delta_days
        g['delta_n2'] = g['Residual_N2_percent'].diff() / time_delta_days
        
        # Fill first observation per well with 0
        g['delta_temp_f'] = g['delta_temp_f'].fillna(0)
        g['delta_ratio'] = g['delta_ratio'].fillna(0)
        g['delta_n2'] = g['delta_n2'].fillna(0)
        
        # 2. Align 7-day ahead target (t + 7 days) using merge_asof per well
        g_target = g[['Date_Sampled', 'Risk_Category', 'Temperature_F']].copy()
        g_target = g_target.rename(columns={
            'Risk_Category': 'target_risk_7d',
            'Temperature_F': 'future_temp_7d',
            'Date_Sampled': 'future_date'
        })
        
        g['ideal_7d'] = g['Date_Sampled'] + pd.Timedelta(days=7)
        
        # Match nearest observation between 4.5 and 9.5 days into the future
        merged = pd.merge_asof(
            g,
            g_target,
            left_on='ideal_7d',
            right_on='future_date',
            tolerance=pd.Timedelta(days=2.5),
            direction='nearest'
        )
        engineered_groups.append(merged)
        
    df_expB = pd.concat(engineered_groups, ignore_index=True)
    
    # Drop rows without a valid 7-day future target (e.g. final week of each well)
    df_expB = df_expB.dropna(subset=['target_risk_7d']).copy()
    
    # Sort chronologically by observation date (t)
    df_expB = df_expB.sort_values(by=['Date_Sampled', 'Well_Name']).reset_index(drop=True)
    
    feature_cols = [
        'CH4_percent', 'CO2_percent', 'O2_percent', 'Residual_N2_percent',
        'Ratio_CH4_CO2', 'Temperature_F', 'Temperature_C',
        'delta_temp_f', 'delta_ratio', 'delta_n2'
    ]
    target_col = 'target_risk_7d'
    
    train_mask = df_expB['Date_Sampled'] < '2015-07-01'
    val_mask = (df_expB['Date_Sampled'] >= '2015-07-01') & (df_expB['Date_Sampled'] < '2016-03-01')
    test_mask = df_expB['Date_Sampled'] >= '2016-03-01'
    
    X_train = df_expB.loc[train_mask, feature_cols]
    y_train = df_expB.loc[train_mask, target_col]
    
    X_val = df_expB.loc[val_mask, feature_cols]
    y_val = df_expB.loc[val_mask, target_col]
    
    X_test = df_expB.loc[test_mask, feature_cols]
    y_test = df_expB.loc[test_mask, target_col]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'full_df': df_expB
    }

if __name__ == "__main__":
    df = load_bridgeton_data()
    data_A = build_experiment_A_data(df)
    print("Experiment A Train shape:", data_A['X_train'].shape, "Test shape:", data_A['X_test'].shape)
    data_B = build_experiment_B_data(df)
    print("Experiment B Train shape:", data_B['X_train'].shape, "Test shape:", data_B['X_test'].shape)
