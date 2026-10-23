"""
test_model_and_inference.py - Tests for model serialization, prediction schema, and reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_bridgeton_raw
from src.preprocessing import prepare_chronological_split
from src.train_model import train_models_on_split, save_models_and_metadata
from src.predict import predict_single, predict_batch, get_model_and_metadata, SCIENTIFIC_DISCLAIMER


def test_train_and_serialize_models(tmp_path):
    """Test model training, evaluation, and saving to disk."""
    df = load_bridgeton_raw()
    split = prepare_chronological_split(df)
    
    # Train models
    out = train_models_on_split(split, seed=42)
    assert 'random_forest' in out['models']
    assert 'logistic_regression' in out['models']
    assert 'dummy' in out['models']
    
    # Save to temp path
    save_models_and_metadata(out['models'], out['results'], out_dir=tmp_path)
    assert (tmp_path / "three_gas_rf_model.joblib").exists()
    assert (tmp_path / "three_gas_lr_model.joblib").exists()
    assert (tmp_path / "dummy_classifier.joblib").exists()
    assert (tmp_path / "model_metadata.json").exists()


def test_predict_single_output_schema():
    """Test predict_single returns required fields and expected disclaimer."""
    # Test typical normal sample
    res_normal = predict_single(52.0, 40.0, 0.5)
    assert res_normal['success'] is True
    assert res_normal['predicted_category'] in ['Low/Normal', 'Medium', 'High']
    assert "Estimated temperature-derived category:" in res_normal['formatted_label']
    assert res_normal['confidence'] is not None
    assert 0.0 <= res_normal['confidence'] <= 1.0
    assert 'CH4_percent' in res_normal['input_measurements']
    assert res_normal['disclaimer'] == SCIENTIFIC_DISCLAIMER

    # Test typical high risk reaction sample (low CH4, high CO2)
    res_high = predict_single(15.0, 65.0, 0.2)
    assert res_high['success'] is True
    assert res_high['predicted_category'] == 'High'


def test_predict_batch_output_schema():
    """Test predict_batch returns enriched DataFrame."""
    batch_df = pd.DataFrame({
        'CH4': [52.0, 15.0],
        'CO2': [40.0, 65.0],
        'O2': [0.5, 0.2]
    })
    success, df_pred, errs = predict_batch(batch_df)
    assert success is True
    assert 'Predicted_Risk_Category' in df_pred.columns
    assert 'Formatted_Prediction' in df_pred.columns
    assert 'Confidence' in df_pred.columns
    assert len(df_pred) == 2


def test_prediction_reproducibility():
    """Verify deterministic predictions across repeated calls."""
    res1 = predict_single(52.0, 40.0, 0.5, model_type='random_forest')
    res2 = predict_single(52.0, 40.0, 0.5, model_type='random_forest')
    assert res1['predicted_category'] == res2['predicted_category']
    assert np.isclose(res1['confidence'], res2['confidence'], atol=1e-5)
