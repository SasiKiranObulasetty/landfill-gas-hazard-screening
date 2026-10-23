"""
test_validation.py - Tests for gas measurement validation and edge-case handling.
"""

import pytest
import pandas as pd
from src.preprocessing import validate_single_sample, validate_batch_dataframe


def test_validate_single_sample_valid():
    """Valid inputs must pass validation."""
    valid, err, warn = validate_single_sample(52.0, 40.0, 0.5)
    assert valid is True
    assert err is None


def test_validate_single_sample_negative():
    """Negative gas concentrations must be rejected."""
    valid, err, warn = validate_single_sample(-5.0, 40.0, 0.5)
    assert valid is False
    assert "Negative concentration" in err


def test_validate_single_sample_excessive_ch4():
    """Concentrations exceeding 100% must be rejected."""
    valid, err, warn = validate_single_sample(105.0, 10.0, 0.5)
    assert valid is False
    assert "exceeds maximum physical limit" in err


def test_validate_single_sample_excessive_o2():
    """Oxygen exceeding atmospheric limit of 25% must be rejected."""
    valid, err, warn = validate_single_sample(50.0, 30.0, 26.5)
    assert valid is False
    assert "exceeds maximum atmospheric air oxygen level" in err


def test_validate_single_sample_excessive_sum():
    """Gas total exceeding 102% must be rejected."""
    valid, err, warn = validate_single_sample(60.0, 50.0, 5.0)  # sum = 115%
    assert valid is False
    assert "exceeds 100%" in err


def test_validate_single_sample_non_numeric():
    """Non-numeric values must be rejected."""
    valid, err, warn = validate_single_sample("invalid", 40.0, 0.5)
    assert valid is False
    assert "numeric" in err


def test_validate_batch_dataframe_valid():
    """Batch DataFrame with valid columns must pass validation."""
    df = pd.DataFrame({
        'CH4': [50.0, 30.0, 15.0],
        'CO2': [40.0, 55.0, 65.0],
        'O2': [0.5, 1.2, 0.2]
    })
    valid, df_clean, errs = validate_batch_dataframe(df)
    assert valid is True
    assert len(errs) == 0
    assert list(df_clean.columns) == ['CH4_percent', 'CO2_percent', 'O2_percent']


def test_validate_batch_dataframe_missing_col():
    """Batch DataFrame missing required column must fail."""
    df = pd.DataFrame({
        'CH4': [50.0, 30.0],
        'CO2': [40.0, 55.0]
    })
    valid, df_clean, errs = validate_batch_dataframe(df)
    assert valid is False
    assert any("Missing required gas columns" in e for e in errs)
