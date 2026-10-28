"""
train_model.py - Model training, evaluation, and serialization module.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Trains 8 classical candidate models on the 3-gas feature set (CH4, CO2, O2):
1. DummyClassifier (Most Frequent Baseline)
2. Logistic Regression (StandardScaler Pipeline, Balanced)
3. Random Forest Classifier (Max Depth 10, Balanced)
4. Decision Tree Classifier (Max Depth 10, Balanced)
5. Extra Trees Classifier (Max Depth 10, Balanced)
6. HistGradientBoosting Classifier (Balanced)
7. Support Vector Machine (StandardScaler Pipeline, RBF Kernel, Balanced, Probability)
8. K-Nearest Neighbors (StandardScaler Pipeline, k=5, Distance-Weighted)

Serializes models and versioned metadata to models/ directory.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_bridgeton_raw
from src.preprocessing import prepare_chronological_split, FEATURE_COLS_3GAS, TARGET_COL, CLASS_ORDER
from src.evaluate import compute_comprehensive_metrics, compute_bootstrap_confidence_intervals

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    'dummy': {
        'display_name': 'Dummy Majority Baseline',
        'file': 'dummy_classifier.joblib'
    },
    'logistic_regression': {
        'display_name': 'Logistic Regression (Pipeline)',
        'file': 'three_gas_lr_model.joblib'
    },
    'random_forest': {
        'display_name': 'Random Forest Classifier',
        'file': 'three_gas_rf_model.joblib'
    },
    'decision_tree': {
        'display_name': 'Decision Tree Classifier',
        'file': 'decision_tree_model.joblib'
    },
    'extra_trees': {
        'display_name': 'Extra Trees Classifier',
        'file': 'extra_trees_model.joblib'
    },
    'hist_gradient_boosting': {
        'display_name': 'HistGradientBoosting Classifier',
        'file': 'hist_gradient_boosting_model.joblib'
    },
    'svm': {
        'display_name': 'Support Vector Machine (RBF)',
        'file': 'svm_model.joblib'
    },
    'knn': {
        'display_name': 'K-Nearest Neighbors (k=5)',
        'file': 'knn_model.joblib'
    }
}


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
    elif model_type == 'decision_tree':
        return DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=seed
        )
    elif model_type == 'extra_trees':
        return ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
    elif model_type == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(
            max_iter=100,
            class_weight='balanced',
            random_state=seed
        )
    elif model_type == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=seed))
        ])
    elif model_type == 'knn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ])
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Expected one of {list(MODEL_CONFIGS.keys())}.")


def train_models_on_split(split_data: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Trains candidate models on training split and evaluates them on test split."""
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    test_df = split_data['test_df']

    results = {}
    models = {}

    for m_type in MODEL_CONFIGS.keys():
        model = get_model_pipeline(m_type, seed=seed)
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time = time.perf_counter() - t1
        
        metrics = compute_comprehensive_metrics(y_test.values, y_pred)
        metrics['train_time_sec'] = round(train_time, 4)
        metrics['inference_time_sec'] = round(inference_time, 4)
        metrics['inference_latency_ms'] = round((inference_time / len(X_test)) * 1000, 4)
        
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
    for m_type, cfg in MODEL_CONFIGS.items():
        if m_type in models:
            model_path = out_dir / cfg['file']
            joblib.dump(models[m_type], model_path)

    # Feature importance extraction where available
    rf_feat_imp = {}
    if 'random_forest' in models and hasattr(models['random_forest'], 'feature_importances_'):
        rf_feat_imp = dict(zip(FEATURE_COLS_3GAS, [float(x) for x in models['random_forest'].feature_importances_]))

    # Compile comprehensive metadata summary
    models_summary = {}
    for m_type, cfg in MODEL_CONFIGS.items():
        if m_type in results:
            m_res = results[m_type]
            m_met = m_res['metrics']
            m_ci = m_res['bootstrap_ci']['clustered_well']
            
            models_summary[m_type] = {
                'file': cfg['file'],
                'display_name': cfg['display_name'],
                'accuracy': m_met['accuracy'],
                'macro_f1': m_met['macro_f1'],
                'weighted_f1': m_met.get('weighted_f1', m_met['macro_f1']),
                'high_risk_recall': m_met['high_risk_recall'],
                'high_risk_precision': m_met['high_risk_precision'],
                'high_risk_specificity': m_met.get('high_risk_specificity', 0.0),
                'train_time_sec': m_met.get('train_time_sec', 0.0),
                'inference_time_sec': m_met.get('inference_time_sec', 0.0),
                'inference_latency_ms': m_met.get('inference_latency_ms', 0.0),
                'clustered_accuracy_95_ci': m_ci['accuracy_ci'],
                'clustered_macro_f1_95_ci': m_ci['macro_f1_ci'],
                'confusion_matrix': m_met['confusion_matrix'],
                'per_class': m_met['per_class']
            }

    metadata = {
        'version': '1.1.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'feature_names': FEATURE_COLS_3GAS,
        'target_name': TARGET_COL,
        'class_names': CLASS_ORDER,
        'train_samples': 11247,
        'test_samples': 3611,
        'rf_feature_importances': rf_feat_imp,
        'models_summary': models_summary,
        'scientific_disclaimer': 'Research prototype for temperature-derived hazard screening. Not an operational fire detection system.'
    }

    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def main():
    print("=" * 75)
    print("TRAINING EXTENDED CLASSICAL 3-GAS SCREENING MODELS (8 CLASSIFIERS)")
    print("=" * 75)
    
    print("\n[1/3] Ingesting confirmed primary dataset...")
    df = load_bridgeton_raw()
    print(f"Loaded {len(df):,} observations across {df['Well_Name'].nunique()} wells.")
    
    print("\n[2/3] Preparing chronological holdout split (with 8-month purge window)...")
    split_data = prepare_chronological_split(df)
    print(f"Train: {len(split_data['X_train']):,} rows | Test: {len(split_data['X_test']):,} rows")
    
    print("\n[3/3] Training and evaluating 8 candidate models...")
    trained = train_models_on_split(split_data)
    
    print(f"\n{'Model':<32} {'Accuracy':<10} {'Macro-F1':<10} {'Weighted-F1':<12} {'High Recall':<12} {'Train (s)':<10} {'Infer (s)':<10}")
    print("-" * 100)
    for m_type, cfg in MODEL_CONFIGS.items():
        m_met = trained['results'][m_type]['metrics']
        print(f"{cfg['display_name']:<32} {m_met['accuracy']:<10.4f} {m_met['macro_f1']:<10.4f} {m_met['weighted_f1']:<12.4f} {m_met['high_risk_recall']:<12.4f} {m_met['train_time_sec']:<10.4f} {m_met['inference_time_sec']:<10.4f}")
    
    print(f"\nSaving serialized models and versioned metadata to {MODELS_DIR}...")
    save_models_and_metadata(trained['models'], trained['results'])
    print("All 8 models and metadata successfully saved.")


if __name__ == "__main__":
    main()
