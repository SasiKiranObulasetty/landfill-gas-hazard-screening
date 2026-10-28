"""
test_model_and_inference.py - Tests for model serialization, prediction schema, and reproducibility.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_bridgeton_raw
from src.preprocessing import prepare_chronological_split
from src.train_model import train_models_on_split, save_models_and_metadata, MODEL_CONFIGS
from src.predict import predict_single, predict_batch, get_model_and_metadata, SCIENTIFIC_DISCLAIMER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_MODEL_KEYS = list(MODEL_CONFIGS.keys())


def test_train_and_serialize_models(tmp_path):
    """Test model training, evaluation, and saving to disk for all 8 models."""
    df = load_bridgeton_raw()
    split = prepare_chronological_split(df)
    
    # Train models
    out = train_models_on_split(split, seed=42)
    for m_key in ALL_MODEL_KEYS:
        assert m_key in out['models'], f"Model {m_key} missing from training output!"
        assert m_key in out['results'], f"Results for {m_key} missing from output!"
        assert 'weighted_f1' in out['results'][m_key]['metrics']
        assert 'train_time_sec' in out['results'][m_key]['metrics']
        assert 'inference_time_sec' in out['results'][m_key]['metrics']
    
    # Save to temp path
    save_models_and_metadata(out['models'], out['results'], out_dir=tmp_path)
    for m_key, cfg in MODEL_CONFIGS.items():
        assert (tmp_path / cfg['file']).exists(), f"File {cfg['file']} for {m_key} was not saved!"
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


def test_all_eight_models_predict_valid_classes():
    """Verify that all 8 candidate classifiers load and produce valid predictions."""
    batch_df = pd.DataFrame({
        'CH4': [52.0, 15.0, 40.0],
        'CO2': [40.0, 65.0, 50.0],
        'O2': [0.5, 0.2, 1.0]
    })
    
    for m_key in ALL_MODEL_KEYS:
        # Single prediction
        res = predict_single(52.0, 40.0, 0.5, model_type=m_key)
        assert res['success'] is True, f"predict_single failed for {m_key}"
        assert res['predicted_category'] in ['Low/Normal', 'Medium', 'High']
        assert res['model_type'] == m_key

        # Batch prediction
        success, df_pred, errs = predict_batch(batch_df, model_type=m_key)
        assert success is True, f"predict_batch failed for {m_key}: {errs}"
        assert len(df_pred) == 3
        for pred in df_pred['Predicted_Risk_Category']:
            assert pred in ['Low/Normal', 'Medium', 'High']


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


def test_model_metadata_json_completeness():
    """Verify saved metadata contains all 8 models with full metrics and timings."""
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    assert meta_path.exists(), "models/model_metadata.json does not exist!"
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    summary = meta.get('models_summary', {})
    for m_key in ALL_MODEL_KEYS:
        assert m_key in summary, f"Model {m_key} missing from metadata summary!"
        entry = summary[m_key]
        for field in ['accuracy', 'macro_f1', 'weighted_f1', 'high_risk_recall', 'high_risk_precision', 'train_time_sec', 'inference_time_sec', 'confusion_matrix', 'per_class']:
            assert field in entry, f"Field '{field}' missing from {m_key} summary!"
            assert entry[field] is not None
