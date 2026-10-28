"""
predict.py - Inference engine and input validation wrapper.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Provides safe, audited predictions:
- Validates input ranges and mass-balance totals
- Rejects non-numeric, negative, or physically impossible inputs
- Injects standard scientific disclaimers
- Formats labels as "Estimated temperature-derived category: <Class>"
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    validate_single_sample,
    validate_batch_dataframe,
    FEATURE_COLS_3GAS,
    CLASS_ORDER
)

MODELS_DIR = PROJECT_ROOT / "models"

SCIENTIFIC_DISCLAIMER = (
    "NOTICE: This model output is an estimated temperature-derived subsurface risk category "
    "based on empirical surface gas measurements (CH4, CO2, O2) from the Bridgeton Landfill quarry. "
    "It is NOT a direct downhole temperature measurement, a confirmed surface fire event, "
    "or an operational safety guarantee. Measurements outside the historical training distribution "
    "may yield unreliable estimates."
)

_MODEL_CACHE: Dict[str, Any] = {}
_METADATA_CACHE: Optional[Dict[str, Any]] = None


def get_model_and_metadata(model_type: str = 'random_forest') -> Tuple[Any, Dict[str, Any]]:
    """Loads and caches the requested trained model and versioned metadata."""
    global _METADATA_CACHE
    
    if _METADATA_CACHE is None:
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                _METADATA_CACHE = json.load(f)
        else:
            _METADATA_CACHE = {
                'version': '1.0.0-dev',
                'models_summary': {}
            }

    file_mapping = {
        'random_forest': 'three_gas_rf_model.joblib',
        'logistic_regression': 'three_gas_lr_model.joblib',
        'dummy': 'dummy_classifier.joblib',
        'decision_tree': 'decision_tree_model.joblib',
        'extra_trees': 'extra_trees_model.joblib',
        'hist_gradient_boosting': 'hist_gradient_boosting_model.joblib',
        'svm': 'svm_model.joblib',
        'knn': 'knn_model.joblib'
    }

    if model_type not in file_mapping:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(file_mapping.keys())}.")

    if model_type not in _MODEL_CACHE:
        model_file = MODELS_DIR / file_mapping[model_type]
        if not model_file.exists():
            # If not yet trained, train on demand
            from src.train_model import main as run_training
            run_training()

        try:
            _MODEL_CACHE[model_type] = joblib.load(model_file)
        except Exception:
            # Fallback if unpickling fails due to environment or version differences
            from src.train_model import main as run_training
            run_training()
            _MODEL_CACHE[model_type] = joblib.load(model_file)

    return _MODEL_CACHE[model_type], _METADATA_CACHE


def predict_single(
    ch4: float,
    co2: float,
    o2: float,
    model_type: str = 'random_forest'
) -> Dict[str, Any]:
    """
    Performs inference for a single gas sample with rigorous validation.
    Returns structured results with explicit temperature-derived category labels.
    """
    is_valid, error_msg, warning_msg = validate_single_sample(ch4, co2, o2)
    if not is_valid:
        return {
            'success': False,
            'error': error_msg,
            'disclaimer': SCIENTIFIC_DISCLAIMER
        }

    model, meta = get_model_and_metadata(model_type)
    
    # Construct 1x3 feature dataframe with correct column names
    X = pd.DataFrame([[float(ch4), float(co2), float(o2)]], columns=FEATURE_COLS_3GAS)
    
    pred_raw = model.predict(X)[0]
    pred_label = f"Estimated temperature-derived category: {pred_raw}"
    
    probs = {}
    confidence = None
    if hasattr(model, 'predict_proba'):
        prob_arr = model.predict_proba(X)[0]
        classes = model.classes_
        probs = {str(c): float(p) for c, p in zip(classes, prob_arr)}
        confidence = float(np.max(prob_arr))

    balance_gas = max(0.0, 100.0 - (float(ch4) + float(co2) + float(o2)))

    return {
        'success': True,
        'predicted_category': pred_raw,
        'formatted_label': pred_label,
        'confidence': confidence,
        'class_probabilities': probs,
        'input_measurements': {
            'CH4_percent': float(ch4),
            'CO2_percent': float(co2),
            'O2_percent': float(o2),
            'Computed_Balance_percent': round(balance_gas, 2)
        },
        'model_type': model_type,
        'model_version': meta.get('version', '1.0.0'),
        'warning': warning_msg,
        'disclaimer': SCIENTIFIC_DISCLAIMER
    }


def predict_batch(
    df_input: pd.DataFrame,
    model_type: str = 'random_forest'
) -> Tuple[bool, pd.DataFrame, List[str]]:
    """
    Performs batch inference on an uploaded DataFrame.
    Validates all inputs, appends predictions and probabilities, and returns (success, df_out, error_log).
    """
    is_valid, cleaned_df, errors = validate_batch_dataframe(df_input)
    if not is_valid:
        return False, df_input, errors

    model, meta = get_model_and_metadata(model_type)
    
    X = cleaned_df[FEATURE_COLS_3GAS]
    preds = model.predict(X)
    
    df_out = cleaned_df.copy()
    df_out['Predicted_Risk_Category'] = preds
    df_out['Formatted_Prediction'] = [f"Estimated temperature-derived category: {p}" for p in preds]

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        classes = model.classes_
        for idx, c in enumerate(classes):
            df_out[f'Prob_{c}'] = np.round(probs[:, idx], 4)
        df_out['Confidence'] = np.round(np.max(probs, axis=1), 4)

    return True, df_out, []
