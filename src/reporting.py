"""
reporting.py - Research reporting utilities and literature comparative table generation.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Compiles verified findings, research paper reviews, and comprehensive metrics
into supervisor-ready reporting artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_verified_literature_review() -> List[Dict[str, str]]:
    """
    Returns authentic research papers reviewed during the project,
    preserving real titles, methods, datasets, and research gaps.
    """
    papers = [
        {
            "Paper": "Paper 1",
            "Title": "A Comprehensive Framework for Risk Probability Assessment of Landfill Fire Incidents Using Fuzzy Fault Tree Analysis",
            "Method": "Fuzzy Fault Tree Analysis (FTA)",
            "Dataset": "Field operational logs & expert elicitation",
            "Key Findings": "Identifies root causes of surface/subsurface landfill fires; models cause-and-effect failure paths under uncertainty.",
            "Research Gap": "No real-time machine learning hotspot detection; no integrated gas-thermal inversion model."
        },
        {
            "Paper": "Paper 4",
            "Title": "A High-Resolution Satellite Survey of Methane Emissions from Sixty North American Municipal Solid Waste Landfills",
            "Method": "High-resolution satellite spectrometry & point-source plume inversion",
            "Dataset": "Satellite survey over 60 North American municipal solid waste landfills",
            "Key Findings": "Reveals large discrepancies between reported facility inventories and direct satellite emission measurements.",
            "Research Gap": "Cannot detect subsurface smoldering; no downhole thermal risk prediction; coverage limited by cloud cover."
        },
        {
            "Paper": "Paper 5",
            "Title": "Monitoring Open Landfill Fires Using Integrated UAV and Sentinel-2 Satellite Imagery",
            "Method": "Random Forest & Gradient Tree Boosting on multispectral imagery",
            "Dataset": "2023 Sarimukti landfill fire event (Indonesia)",
            "Key Findings": "Achieves ~81% classification accuracy in delineating post-fire surface burn scars and mitigation priorities.",
            "Research Gap": "Focuses on post-fire mapping rather than pre-fire prediction; no subsurface gas monitoring."
        },
        {
            "Paper": "Paper 6",
            "Title": "Optimizing LandGEM Model Parameters Using a Machine Learning Method to Improve Landfill Methane Generation Estimates",
            "Method": "Inverse modeling & K-Nearest Neighbors (KNN)",
            "Dataset": "EPA LandGEM nationwide US landfill databases",
            "Key Findings": "Demonstrates that ML parameter tuning reduces methane generation estimate errors compared to default AP-42 parameters.",
            "Research Gap": "Predicts macro-scale gas volume over decades; cannot detect localized smoldering reaction fronts."
        },
        {
            "Paper": "Paper 7",
            "Title": "Innovative Drone-Based Methodology for Quantifying Methane Emissions from Landfills",
            "Method": "UAV mass-balance aerial sampling & chamber comparison",
            "Dataset": "Aerial transects at an Italian municipal landfill",
            "Key Findings": "Drone sensor arrays provide flexible, rapid screening of surface methane plumes.",
            "Research Gap": "Short measurement campaigns; no subsurface thermal modeling; no fire risk forecasting."
        },
        {
            "Paper": "Paper 10",
            "Title": "Genetic Algorithm–Artificial Neural Network (GA-ANN) Modeling for VOC Emission Rates from Landfills",
            "Method": "Hybrid Genetic Algorithm & Artificial Neural Network",
            "Dataset": "9 months of field data (105 sets) from a Beijing landfill",
            "Key Findings": "Demonstrates superior non-linear modeling of volatile emissions driven by weather and waste composition.",
            "Research Gap": "Limited to surface air quality; no thermal reaction front modeling or subsurface hazard classification."
        },
        {
            "Paper": "Paper 25 / k6",
            "Title": "Subsurface Landfill Fire Prediction Using Machine Learning Approaches (Sabrin, 2018)",
            "Method": "SVM, ANN, Random Forest, Decision Trees",
            "Dataset": "Bridgeton Landfill MDNR compliance monitoring records (2013–2016)",
            "Key Findings": "Formulates temperature-derived risk thresholds (<131 F, 131-176 F, >=176 F); reports high classification accuracy.",
            "Research Gap": "Trained on synthetic temperature cutoffs rather than confirmed fire events; 7-day forecasting failed without valve telemetry."
        }
    ]
    return papers


def get_model_recalculation_summary() -> pd.DataFrame:
    """Returns the recalculated metrics table from the Phase 10 audit."""
    data = [
        {
            "Model": "Dummy Majority Baseline",
            "Features": "None (Prior Class Distribution)",
            "Split": "Chronological (N=3,611)",
            "Accuracy": 0.6336,
            "Macro_F1": 0.2586,
            "High_Risk_Recall": 0.0000,
            "Clustered_Acc_95_CI": "[0.6120, 0.6550]"
        },
        {
            "Model": "Logistic Regression (Pipeline)",
            "Features": "CH4, CO2, O2 (3 Gases)",
            "Split": "Chronological (N=3,611)",
            "Accuracy": 0.9272,
            "Macro_F1": 0.9301,
            "High_Risk_Recall": 0.9984,
            "Clustered_Acc_95_CI": "[0.9179, 0.9349]"
        },
        {
            "Model": "Random Forest (Production)",
            "Features": "CH4, CO2, O2 (3 Gases)",
            "Split": "Chronological (N=3,611)",
            "Accuracy": 0.9601,
            "Macro_F1": 0.9606,
            "High_Risk_Recall": 1.0000,
            "Clustered_Acc_95_CI": "[0.9542, 0.9662]"
        },
        {
            "Model": "Random Forest (Spatio-Temporal)",
            "Features": "CH4, CO2, O2 (3 Gases)",
            "Split": "30 Unseen Wells + Future Time (N=704)",
            "Accuracy": 0.9574,
            "Macro_F1": 0.9591,
            "High_Risk_Recall": 1.0000,
            "Clustered_Acc_95_CI": "[0.9435, 0.9709]"
        },
        {
            "Model": "Random Forest (GroupKFold)",
            "Features": "CH4, CO2, O2 (3 Gases)",
            "Split": "5-Fold Grouped Well CV (N=18,469)",
            "Accuracy": 0.9567,
            "Macro_F1": 0.9551,
            "High_Risk_Recall": 0.9997,
            "Clustered_Acc_95_CI": "[0.9542, 0.9592]"
        },
        {
            "Model": "7-Day Forecasting RF",
            "Features": "7 Gas Features at Time t",
            "Split": "Chronological t+7d (N=1,391)",
            "Accuracy": 0.4579,
            "Macro_F1": 0.2908,
            "High_Risk_Recall": 0.0811,
            "Clustered_Acc_95_CI": "[0.4012, 0.5146]"
        }
    ]
    return pd.DataFrame(data)


def get_classical_model_comparison_summary() -> pd.DataFrame:
    """
    Returns the comprehensive 8-model classical machine learning comparison table
    evaluated on the exact chronological test split (N=3,611).
    Reads live results from models/model_metadata.json.
    """
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        summary = meta.get('models_summary', {})
        rows = []
        for key, info in summary.items():
            acc_ci = info.get('clustered_accuracy_95_ci', [0.0, 0.0])
            f1_ci = info.get('clustered_macro_f1_95_ci', [0.0, 0.0])
            rows.append({
                "Model_Key": key,
                "Model": info.get('display_name', key),
                "Split": "Chronological (N=3,611)",
                "Accuracy": round(info.get('accuracy', 0.0), 4),
                "Macro_F1": round(info.get('macro_f1', 0.0), 4),
                "Weighted_F1": round(info.get('weighted_f1', 0.0), 4),
                "High_Risk_Recall": round(info.get('high_risk_recall', 0.0), 4),
                "High_Risk_Precision": round(info.get('high_risk_precision', 0.0), 4),
                "High_Risk_Specificity": round(info.get('high_risk_specificity', 0.0), 4),
                "Train_Time_Sec": round(info.get('train_time_sec', 0.0), 4),
                "Infer_Time_Sec": round(info.get('inference_time_sec', 0.0), 4),
                "Clustered_Acc_95_CI": f"[{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]" if len(acc_ci) == 2 else "N/A",
                "Clustered_Macro_F1_95_CI": f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]" if len(f1_ci) == 2 else "N/A"
            })
        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame()

