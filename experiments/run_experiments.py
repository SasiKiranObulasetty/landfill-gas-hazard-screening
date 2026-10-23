"""
run_experiments.py - Unified Modeling & Evaluation Pipeline
Compares Experiment A (Gas-Only Inversion) and Experiment B (7-Day Lead Forecasting).
Trains DummyClassifier, LogisticRegression, and RandomForestClassifier.
Generates metrics JSON, comparison CSV, and publication-grade evaluation plots.
"""

import os
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)

from preprocessing import load_bridgeton_data, build_experiment_A_data, build_experiment_B_data

EXPERIMENTS_DIR = Path(r"c:\Users\obula\OneDrive\Desktop\4th_year_project\experiments")
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ['Low/Normal', 'Medium', 'High']

def plot_confusion_matrix(cm, classes, title, save_path, cmap=plt.cm.Blues):
    """Generates and saves a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True Label (Regulatory)',
        xlabel='Predicted Label (Model)'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved confusion matrix: {save_path.name}")

def plot_feature_importance(importances, feature_names, title, save_path):
    """Plots and saves Random Forest feature importances."""
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(sorted_importances)), sorted_importances[::-1], color='#1f77b4', align='center')
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(sorted_features[::-1], fontsize=10)
    ax.set_xlabel('Gini Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved feature importance: {save_path.name}")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, exp_name):
    """Fits model and evaluates performance metrics against test split."""
    print(f"\nEvaluating {model_name} on {exp_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    report_dict = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    
    # Save confusion matrix plot
    plot_filename = f"cm_{exp_name.lower()}_{model_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(
        cm, CLASS_ORDER,
        f"{model_name} - {exp_name}\nOut-of-Time Test Set",
        RESULTS_DIR / plot_filename
    )
    
    metrics = {
        'model_name': model_name,
        'experiment': exp_name,
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'weighted_f1': round(float(weighted_f1), 4),
        'macro_precision': round(float(macro_prec), 4),
        'macro_recall': round(float(macro_rec), 4),
        'class_low_f1': round(float(report_dict['Low/Normal']['f1-score']), 4),
        'class_med_f1': round(float(report_dict['Medium']['f1-score']), 4),
        'class_high_f1': round(float(report_dict['High']['f1-score']), 4),
        'class_low_rec': round(float(report_dict['Low/Normal']['recall']), 4),
        'class_med_rec': round(float(report_dict['Medium']['recall']), 4),
        'class_high_rec': round(float(report_dict['High']['recall']), 4),
        'class_low_prec': round(float(report_dict['Low/Normal']['precision']), 4),
        'class_med_prec': round(float(report_dict['Medium']['precision']), 4),
        'class_high_prec': round(float(report_dict['High']['precision']), 4),
        'confusion_matrix': cm.tolist(),
        'test_support_total': int(len(y_test)),
        'test_class_counts': {k: int(report_dict[k]['support']) for k in CLASS_ORDER}
    }
    
    print(f"Results for {model_name} [{exp_name}]:")
    print(f"  Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")
    print(f"  High Class Recall: {report_dict['High']['recall']:.4f} | Precision: {report_dict['High']['precision']:.4f}")
    
    return metrics, model

def main():
    print("=" * 70)
    print("STARTING BRIDGETON LANDFILL MACHINE LEARNING EXPERIMENTS")
    print("=" * 70)
    
    # 1. Load data
    df = load_bridgeton_data()
    
    # Check class imbalance in full dataset
    class_counts = df['Risk_Category'].value_counts()
    print("\nFull Dataset Class Distribution:")
    for k in CLASS_ORDER:
        print(f"  {k}: {class_counts[k]} ({class_counts[k]/len(df)*100:.2f}%)")
        
    all_results = []
    
    # =========================================================
    # EXPERIMENT A: Gas-Only Contemporaneous Classification
    # =========================================================
    print("\n" + "=" * 50)
    print("EXPERIMENT A: GAS-ONLY CONTEMPORANEOUS CLASSIFICATION")
    print("Excluding Temperature to evaluate chemical reaction detection")
    print("=" * 50)
    
    data_A = build_experiment_A_data(df)
    X_train_A, y_train_A = data_A['X_train'], data_A['y_train']
    X_test_A, y_test_A = data_A['X_test'], data_A['y_test']
    
    print(f"Train samples: {len(X_train_A)}, Test samples: {len(X_test_A)}")
    print(f"Features ({len(data_A['feature_cols'])}): {data_A['feature_cols']}")
    
    # Model A1: Dummy Baseline (Majority class)
    dummy_A = DummyClassifier(strategy='most_frequent')
    m_dummy_A, _ = evaluate_model(dummy_A, X_train_A, y_train_A, X_test_A, y_test_A, "Majority Class Baseline", "ExpA_GasOnly")
    all_results.append(m_dummy_A)
    
    # Model A2: Logistic Regression with StandardScaler
    pipe_lr_A = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    m_lr_A, _ = evaluate_model(pipe_lr_A, X_train_A, y_train_A, X_test_A, y_test_A, "Logistic Regression", "ExpA_GasOnly")
    all_results.append(m_lr_A)
    
    # Model A3: Random Forest Classifier
    rf_A = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)
    m_rf_A, fitted_rf_A = evaluate_model(rf_A, X_train_A, y_train_A, X_test_A, y_test_A, "Random Forest", "ExpA_GasOnly")
    all_results.append(m_rf_A)
    
    # Save Feature Importance Plot for Exp A
    plot_feature_importance(
        fitted_rf_A.feature_importances_,
        data_A['feature_cols'],
        "Experiment A: Random Forest Feature Importance (Gas-Only Inversion)",
        RESULTS_DIR / "feature_importance_expA_gas_only.png"
    )
    
    # =========================================================
    # EXPERIMENT B: Seven-Day-Ahead Risk Forecasting
    # =========================================================
    print("\n" + "=" * 50)
    print("EXPERIMENT B: SEVEN-DAY-AHEAD RISK FORECASTING")
    print("Using strictly historical features to predict future state (t + 7 days)")
    print("=" * 50)
    
    data_B = build_experiment_B_data(df)
    X_train_B, y_train_B = data_B['X_train'], data_B['y_train']
    X_test_B, y_test_B = data_B['X_test'], data_B['y_test']
    
    print(f"Train samples: {len(X_train_B)}, Test samples: {len(X_test_B)}")
    print(f"Features ({len(data_B['feature_cols'])}): {data_B['feature_cols']}")
    
    # Model B1: Dummy Baseline (Majority class)
    dummy_B = DummyClassifier(strategy='most_frequent')
    m_dummy_B, _ = evaluate_model(dummy_B, X_train_B, y_train_B, X_test_B, y_test_B, "Majority Class Baseline", "ExpB_7dLead")
    all_results.append(m_dummy_B)
    
    # Model B2: Logistic Regression with StandardScaler
    pipe_lr_B = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    m_lr_B, _ = evaluate_model(pipe_lr_B, X_train_B, y_train_B, X_test_B, y_test_B, "Logistic Regression", "ExpB_7dLead")
    all_results.append(m_lr_B)
    
    # Model B3: Random Forest Classifier
    rf_B = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)
    m_rf_B, fitted_rf_B = evaluate_model(rf_B, X_train_B, y_train_B, X_test_B, y_test_B, "Random Forest", "ExpB_7dLead")
    all_results.append(m_rf_B)
    
    # Save Feature Importance Plot for Exp B
    plot_feature_importance(
        fitted_rf_B.feature_importances_,
        data_B['feature_cols'],
        "Experiment B: Random Forest Feature Importance (7-Day Lead Forecast)",
        RESULTS_DIR / "feature_importance_expB_lead7d.png"
    )
    
    # =========================================================
    # SAVE STRUCTURED COMPARISON CSV & METRICS JSON
    # =========================================================
    summary_csv_path = RESULTS_DIR / "experiment_comparison.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            'experiment', 'model_name', 'accuracy', 'macro_f1', 'weighted_f1',
            'macro_precision', 'macro_recall',
            'class_low_f1', 'class_med_f1', 'class_high_f1',
            'class_low_rec', 'class_med_rec', 'class_high_rec',
            'class_low_prec', 'class_med_prec', 'class_high_prec',
            'test_support_total'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for res in all_results:
            writer.writerow(res)
            
    print(f"\n[Artifact] Saved comparison CSV to: {summary_csv_path.name}")
    
    summary_json_path = RESULTS_DIR / "metrics_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Artifact] Saved metrics JSON to: {summary_json_path.name}")
    
    print("\nAll experiments, evaluations, and plots completed successfully!")

if __name__ == "__main__":
    main()
