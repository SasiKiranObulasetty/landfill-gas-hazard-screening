"""
train_model.py - Model training, evaluation, and serialization module.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Trains 3 candidate models on the 3-gas feature set (CH4, CO2, O2):
1. DummyClassifier (Most Frequent Baseline)
2. Logistic Regression (StandardScaler Pipeline, Balanced)
3. Random Forest Classifier (Max Depth 10, Balanced)

Serializes models and versioned metadata to models/ directory.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_bridgeton_raw
from src.preprocessing import prepare_chronological_split, FEATURE_COLS_3GAS, TARGET_COL, CLASS_ORDER
from src.evaluate import compute_comprehensive_metrics, compute_bootstrap_confidence_intervals

MODELS_DIR = PROJECT_ROOT / "models"


def get_model_pipeline(model_type: str = 'random_forest', seed: int = 42):
    """Factory function returning configured scikit-learn estimators."""
    if model_type == 'dummy':
        return DummyClassifier(strategy='most_frequent')
    elif model_type == 'logistic_regression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed))
        ])
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'dummy', 'logistic_regression', or 'random_forest'.")


def train_models_on_split(split_data: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Trains candidate models on training split and evaluates them on test split."""
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    test_df = split_data['test_df']

    results = {}
    models = {}

    for m_type in ['dummy', 'logistic_regression', 'random_forest']:
        model = get_model_pipeline(m_type, seed=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = compute_comprehensive_metrics(y_test.values, y_pred)
        boot_ci = compute_bootstrap_confidence_intervals(test_df, y_test.values, y_pred, seed=seed)
        
        models[m_type] = model
        results[m_type] = {
            'metrics': metrics,
            'bootstrap_ci': boot_ci,
            'predictions': y_pred.tolist()
        }

    return {'models': models, 'results': results}


def save_models_and_metadata(
    models: Dict[str, Any],
    results: Dict[str, Any],
    out_dir: Optional[Path] = None
) -> Path:
    """Saves serialized models and metadata JSON to disk."""
    if out_dir is None:
        out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    rf_path = out_dir / "three_gas_rf_model.joblib"
    lr_path = out_dir / "three_gas_lr_model.joblib"
    dummy_path = out_dir / "dummy_classifier.joblib"

    joblib.dump(models['random_forest'], rf_path)
    joblib.dump(models['logistic_regression'], lr_path)
    joblib.dump(models['dummy'], dummy_path)

    # Compile metadata
    rf_feat_imp = {}
    if hasattr(models['random_forest'], 'feature_importances_'):
        rf_feat_imp = dict(zip(FEATURE_COLS_3GAS, [float(x) for x in models['random_forest'].feature_importances_]))

    metadata = {
        'version': '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'feature_names': FEATURE_COLS_3GAS,
        'target_name': TARGET_COL,
        'class_names': CLASS_ORDER,
        'train_samples': 11247,
        'test_samples': 3611,
        'rf_feature_importances': rf_feat_imp,
        'models_summary': {
            'random_forest': {
                'file': 'three_gas_rf_model.joblib',
                'accuracy': results['random_forest']['metrics']['accuracy'],
                'macro_f1': results['random_forest']['metrics']['macro_f1'],
                'high_risk_recall': results['random_forest']['metrics']['high_risk_recall'],
                'high_risk_precision': results['random_forest']['metrics']['high_risk_precision'],
                'clustered_accuracy_95_ci': results['random_forest']['bootstrap_ci']['clustered_well']['accuracy_ci'],
                'clustered_macro_f1_95_ci': results['random_forest']['bootstrap_ci']['clustered_well']['macro_f1_ci']
            },
            'logistic_regression': {
                'file': 'three_gas_lr_model.joblib',
                'accuracy': results['logistic_regression']['metrics']['accuracy'],
                'macro_f1': results['logistic_regression']['metrics']['macro_f1'],
                'high_risk_recall': results['logistic_regression']['metrics']['high_risk_recall'],
                'high_risk_precision': results['logistic_regression']['metrics']['high_risk_precision'],
                'clustered_accuracy_95_ci': results['logistic_regression']['bootstrap_ci']['clustered_well']['accuracy_ci'],
                'clustered_macro_f1_95_ci': results['logistic_regression']['bootstrap_ci']['clustered_well']['macro_f1_ci']
            },
            'dummy': {
                'file': 'dummy_classifier.joblib',
                'accuracy': results['dummy']['metrics']['accuracy'],
                'macro_f1': results['dummy']['metrics']['macro_f1']
            }
        },
        'scientific_disclaimer': 'Research prototype for temperature-derived hazard screening. Not an operational fire detection system.'
    }

    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def main():
    print("="*70)
    print("TRAINING PRODUCTION 3-GAS SCREENING MODELS")
    print("="*70)
    
    print("\n[1/3] Ingesting confirmed primary dataset...")
    df = load_bridgeton_raw()
    print(f"Loaded {len(df):,} observations across {df['Well_Name'].nunique()} wells.")
    
    print("\n[2/3] Preparing chronological holdout split (with 8-month purge window)...")
    split_data = prepare_chronological_split(df)
    print(f"Train: {len(split_data['X_train']):,} rows | Test: {len(split_data['X_test']):,} rows")
    
    print("\n[3/3] Training and evaluating candidate models...")
    trained = train_models_on_split(split_data)
    
    rf_m = trained['results']['random_forest']['metrics']
    lr_m = trained['results']['logistic_regression']['metrics']
    dum_m = trained['results']['dummy']['metrics']
    
    print(f"\n--- EVALUATION METRICS ON TEST SET (N={split_data['X_test'].shape[0]}) ---")
    print(f"Dummy Baseline:       Accuracy = {dum_m['accuracy']:.4f} | Macro-F1 = {dum_m['macro_f1']:.4f}")
    print(f"Logistic Regression:  Accuracy = {lr_m['accuracy']:.4f} | Macro-F1 = {lr_m['macro_f1']:.4f} | High Recall = {lr_m['high_risk_recall']:.4f}")
    print(f"Random Forest (Top):  Accuracy = {rf_m['accuracy']:.4f} | Macro-F1 = {rf_m['macro_f1']:.4f} | High Recall = {rf_m['high_risk_recall']:.4f}")
    
    print(f"\nSaving serialized models and versioned metadata to {MODELS_DIR}...")
    save_models_and_metadata(trained['models'], trained['results'])
    print("Models and metadata successfully saved.")


if __name__ == "__main__":
    main()
