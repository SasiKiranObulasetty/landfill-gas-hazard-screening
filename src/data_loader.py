"""
data_loader.py - Data ingestion and schema validation module.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Loads the primary confirmed Bridgeton Landfill monitoring dataset:
Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx
Preserves source datasets intact. Never modifies original records.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


# Default primary dataset location (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_DATASET_REL = Path("Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx")
FALLBACK_ZIP_REL = Path("Datasets/datasets.zip")

DOCUMENTED_SCHEMA = {
    'Well_Name': {'type': 'object', 'unit': 'Identifier', 'description': 'Gas extraction wellhead identifier (e.g., GEW-001 to GEW-150)'},
    'Date_Sampled': {'type': 'datetime64[ns]', 'unit': 'Timestamp', 'description': 'Sampling date recorded during field monitoring'},
    'CH4_percent': {'type': 'float64', 'unit': 'Vol %', 'description': 'Methane concentration (0.0 to 100.0%) measured via dual-wavelength infrared sensor'},
    'CO2_percent': {'type': 'float64', 'unit': 'Vol %', 'description': 'Carbon dioxide concentration (0.0 to 100.0%) measured via dual-wavelength infrared sensor'},
    'O2_percent': {'type': 'float64', 'unit': 'Vol %', 'description': 'Oxygen concentration (0.0 to 25.0%) measured via electrochemical fuel cell sensor'},
    'Residual_N2_percent': {'type': 'float64', 'unit': 'Vol %', 'description': 'Derived stoichiometric nitrogen proxy: max(0, Balance - 3.76*O2) per 40 CFR 60.754'},
    'Temperature_F': {'type': 'float64', 'unit': 'deg F', 'description': 'Wellhead temperature recorded by downhole thermocouple probe'},
    'Temperature_C': {'type': 'float64', 'unit': 'deg C', 'description': 'Wellhead temperature converted to Celsius ((F-32)*5/9)'},
    'Ratio_CH4_CO2': {'type': 'float64', 'unit': 'Ratio', 'description': 'Methane to carbon dioxide volumetric ratio (CH4 / (CO2 + 1e-6))'},
    'Risk_Category': {'type': 'object', 'unit': 'Categorical', 'description': 'Reconstructed thermal risk category (Low/Normal: <131 F, Medium: 131-176 F, High: >=176 F)'}
}

VALID_RISK_CATEGORIES = ['Low/Normal', 'Medium', 'High']


def resolve_dataset_path(custom_path: Optional[str] = None) -> Path:
    """Resolves dataset path, supporting custom paths, project-relative paths, and fallback zip."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    
    primary = PROJECT_ROOT / PRIMARY_DATASET_REL
    if primary.exists():
        return primary
    
    # Check alternate confirmed paths
    alt = PROJECT_ROOT / "Confirmed_Minimal_Datasets/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx"
    if alt.exists():
        return alt
        
    return primary


def load_bridgeton_raw(custom_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx directly using
    fast, memory-efficient OpenXML parsing without requiring third-party excel drivers.
    Returns a pristine working copy of the dataset.
    """
    file_path = resolve_dataset_path(custom_path)
    
    xlsx_bytes = None
    if file_path.exists() and file_path.suffix == '.xlsx':
        with open(file_path, 'rb') as f:
            xlsx_bytes = f.read()
    else:
        # Fallback to datasets.zip
        zip_path = PROJECT_ROOT / FALLBACK_ZIP_REL
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                # search for match
                matching = [n for n in z.namelist() if 'k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx' in n]
                if matching:
                    xlsx_bytes = z.read(matching[0])
                    
    if xlsx_bytes is None:
        raise FileNotFoundError(
            f"Could not locate Bridgeton dataset at {file_path} or within {FALLBACK_ZIP_REL}."
        )

    # Parse OpenXML sheetData directly
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

    if not rows_data:
        raise ValueError("Spreadsheet sheetData contains 0 rows.")

    col_map = {
        'A': 'Well_Name',
        'B': 'Date_Sampled',
        'C': 'CH4_percent',
        'D': 'CO2_percent',
        'E': 'O2_percent',
        'F': 'Residual_N2_percent',
        'G': 'Temperature_F',
        'H': 'Temperature_C',
        'I': 'Ratio_CH4_CO2',
        'J': 'Risk_Category'
    }

    records = [{colname: r.get(letter, '') for letter, colname in col_map.items()} for r in rows_data[1:]]
    df = pd.DataFrame(records)

    # Cast numeric columns
    num_cols = ['CH4_percent', 'CO2_percent', 'O2_percent', 'Residual_N2_percent', 'Temperature_F', 'Temperature_C', 'Ratio_CH4_CO2']
    for nc in num_cols:
        df[nc] = pd.to_numeric(df[nc], errors='coerce')

    df['Date_Sampled'] = pd.to_datetime(df['Date_Sampled'], errors='coerce')
    df['Well_Name'] = df['Well_Name'].astype(str).str.strip()
    df['Risk_Category'] = df['Risk_Category'].astype(str).str.strip()
    
    # Sort chronologically by Date_Sampled and Well_Name
    df = df.sort_values(by=['Date_Sampled', 'Well_Name']).reset_index(drop=True)
    
    return df


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validates that a dataframe adheres strictly to the documented schema.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    required_cols = list(DOCUMENTED_SCHEMA.keys())
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        
    if 'Risk_Category' in df.columns:
        invalid_cats = set(df['Risk_Category'].dropna().unique()) - set(VALID_RISK_CATEGORIES)
        if invalid_cats:
            issues.append(f"Found unexpected categories in Risk_Category: {invalid_cats}")
            
    if 'CH4_percent' in df.columns:
        neg_ch4 = (df['CH4_percent'] < 0).sum()
        if neg_ch4 > 0:
            issues.append(f"Found {neg_ch4} negative CH4 readings.")
            
    if 'CO2_percent' in df.columns:
        neg_co2 = (df['CO2_percent'] < 0).sum()
        if neg_co2 > 0:
            issues.append(f"Found {neg_co2} negative CO2 readings.")
            
    if 'O2_percent' in df.columns:
        neg_o2 = (df['O2_percent'] < 0).sum()
        if neg_o2 > 0:
            issues.append(f"Found {neg_o2} negative O2 readings.")

    return len(issues) == 0, issues


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive descriptive statistics and metadata for the dataset."""
    total_rows = len(df)
    unique_wells = df['Well_Name'].nunique() if 'Well_Name' in df.columns else 0
    date_min = df['Date_Sampled'].min() if 'Date_Sampled' in df.columns else None
    date_max = df['Date_Sampled'].max() if 'Date_Sampled' in df.columns else None
    
    cat_counts = df['Risk_Category'].value_counts().to_dict() if 'Risk_Category' in df.columns else {}
    cat_pcts = {k: (v / total_rows) * 100 for k, v in cat_counts.items()} if total_rows > 0 else {}
    
    missing_counts = df.isna().sum().to_dict()
    
    return {
        'total_rows': total_rows,
        'unique_wells': unique_wells,
        'date_min': date_min,
        'date_max': date_max,
        'date_range_days': (date_max - date_min).days if (date_min and date_max) else 0,
        'category_counts': cat_counts,
        'category_percentages': cat_pcts,
        'missing_counts': missing_counts,
        'column_info': DOCUMENTED_SCHEMA
    }
