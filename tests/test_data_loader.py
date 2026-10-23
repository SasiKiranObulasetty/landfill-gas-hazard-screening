"""
test_data_loader.py - Automated tests for data loading and schema validation.
"""

import pytest
import pandas as pd
from src.data_loader import (
    load_bridgeton_raw,
    validate_dataset_schema,
    get_dataset_summary,
    DOCUMENTED_SCHEMA,
    VALID_RISK_CATEGORIES
)


def test_load_bridgeton_raw_shape():
    """Verify that the primary dataset loads exactly 18,469 rows and expected columns."""
    df = load_bridgeton_raw()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 18469
    assert 'Well_Name' in df.columns
    assert 'CH4_percent' in df.columns
    assert 'CO2_percent' in df.columns
    assert 'O2_percent' in df.columns
    assert 'Risk_Category' in df.columns


def test_validate_dataset_schema():
    """Verify that the loaded dataset satisfies all schema constraints."""
    df = load_bridgeton_raw()
    is_valid, issues = validate_dataset_schema(df)
    assert is_valid, f"Dataset schema validation failed with issues: {issues}"
    assert len(issues) == 0


def test_get_dataset_summary():
    """Verify dataset summary metadata."""
    df = load_bridgeton_raw()
    summary = get_dataset_summary(df)
    assert summary['total_rows'] == 18469
    assert summary['unique_wells'] == 150
    assert summary['date_min'] is not None
    assert summary['date_max'] is not None
    assert set(summary['category_counts'].keys()) == set(VALID_RISK_CATEGORIES)
