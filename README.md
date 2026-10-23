# AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)
[![Audit Status](https://img.shields.io/badge/Audit-Phase%2010%20Verified-brightgreen.svg)](experiments/phase10_scientific_audit/)

An end-to-end machine learning system for non-invasive classification of subsurface thermal reaction categories in solid waste landfills using routine surface wellhead gas measurements (**Methane**, **Carbon Dioxide**, and **Oxygen**).

Developed for university final-year engineering project demonstration and empirical research evaluation.

---

## 1. Project Purpose and Scientific Scope

### 1.1 Demonstrated Capabilities (Supported by Data)
- **Contemporaneous Thermal State Screening:** Accurately classifies extraction wellheads into temperature-derived risk categories (`Low/Normal`, `Medium`, `High`) using strictly 3 surface gas concentrations ($\text{CH}_4, \text{CO}_2, \text{O}_2$).
- **High-Risk Reaction Detection:** Random Forest achieves **96.01% out-of-sample accuracy** and **100.0% recall on severe thermal reactions** ($T \ge 176^\circ\text{F} / 80^\circ\text{C}$).
- **Spatial Wellhead Localization:** Out-of-sample evaluation on **30 physically unseen extraction wells** confirms **95.74% accuracy and 100.0% High-Risk recall**, demonstrating that the model accurately maps which wells are actively drawing reaction gases.
- **Physical Grounding:** Grounded in thermodynamic methanogenesis cessation: at $T \ge 176^\circ\text{F}$, methanogenic archaea die rapidly, forcing $\text{CH}_4 < 35\%$ while oxidation drives $\text{CO}_2 > 50\%$.

### 1.2 Scientific Boundaries (What the System Does NOT Do)
- **NOT an Open Fire Prediction System:** The dataset records downhole smoldering temperatures; it contains zero confirmed surface flame outbreak incidents or fire department call logs.
- **NOT a 7-Day Forecasting System:** 7-day predictive time-series forecasting fails (**35.79% – 45.79% accuracy**, consistently underperforming the 43.16% – 52.48% naive persistence baselines) because 58.9% of hot wells reverse to normal within 7 days in empirical monitoring records. Operational interventions have been hypothesized in literature, but remain unverified due to the absence of operational records in this dataset.
- **NOT a Certified Safety System:** Designed as an engineering screening prototype for prioritizing wellhead thermocouple inspections, not a life-safety alarm.

---

## 2. System Architecture & Directory Structure

```text
├── app.py                          # Streamlit 7-page interactive research web application
├── requirements.txt                # Versioned Python package dependencies
├── .gitignore                      # Git ignore rules (virtual environments, caches)
├── README.md                       # Master project documentation
│
├── src/                            # Modular engineering source code
│   ├── __init__.py
│   ├── data_loader.py              # Ingestion, caching, and schema validation
│   ├── preprocessing.py            # Feature extraction, leakage prevention, split protocols
│   ├── train_model.py              # Model training, tuning, and serialization
│   ├── predict.py                  # Inference engine, physical validation, disclaimers
│   ├── evaluate.py                 # Metrics, confusion matrix, clustered bootstrap CIs
│   ├── visualizations.py           # Interactive Plotly & publication Matplotlib charts
│   └── reporting.py                # Academic literature reviews and audit summaries
│
├── models/                         # Serialized model artifacts and metadata
│   ├── three_gas_rf_model.joblib   # Production Random Forest classifier (3-gas)
│   ├── three_gas_lr_model.joblib   # Production Logistic Regression pipeline
│   ├── dummy_classifier.joblib     # Majority baseline estimator
│   └── model_metadata.json         # Versioning, feature names, hyperparameter metadata
│
├── tests/                          # Automated test suite
│   ├── __init__.py
│   ├── test_data_loader.py         # Data loading, row count, and schema tests
│   ├── test_preprocessing_and_splits.py # Zero-leakage and split integrity tests
│   ├── test_validation.py          # Input validation, bounds, and total sum tests
│   ├── test_model_and_inference.py # Model serialization and inference schema tests
│   ├── test_evaluation_metrics.py  # Metric calculations and clustered bootstrap tests
│   └── test_app_startup.py         # App syntax and page registration tests
│
├── reports/                        # Research documentation
│   ├── Final_Research_Report.md    # 16-section comprehensive supervisor-ready report
│   └── figures/                    # High-resolution publication figures
│
├── Confirmed_Minimal_Datasets/     # Source monitoring data
│   └── primary/
│       └── k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx # Primary verified dataset
│
└── experiments/                    # Previous research phases and scientific audits
    ├── phase9_validation/          # Independent reproduction and ablation results
    └── phase10_scientific_audit/   # Comprehensive scientific audit artifacts
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Operating System: Windows, macOS, or Linux

### Quick Setup

```bash
# 1. Clone repository (or navigate to workspace)
cd 4th_year_project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the Application

To launch the interactive multi-page research portal:

```bash
streamlit run app.py
```

Open your browser to the local URL (default: `http://localhost:8501`).

### Demonstration Sequence for Supervisor Presentation
1. **Page A (Project Overview):** Review research motivation, facility context (Bridgeton Landfill, 18,469 records), and demonstrated capabilities vs. disclaimers.
2. **Page B (Dataset Explorer):** Demonstrate interactive filtering by well and date, inspect documented schema units, and show the CSV export functionality.
3. **Page C (Risk-Category Classification):** Inspect the 3-gas Random Forest performance (**96.01% accuracy, 1.0000 High-Risk recall**), review the confusion matrix, and explain the feature importance of $\text{CH}_4$ and $\text{CO}_2$.
4. **Page D (Interactive Gas Screening):**
   - *Normal Biogas Case:* Enter $\text{CH}_4 = 52.0\%$, $\text{CO}_2 = 40.0\%$, $\text{O}_2 = 0.5\%$. Output: **Estimated temperature-derived category: Low/Normal**.
   - *Reaction Hotspot Case:* Enter $\text{CH}_4 = 15.0\%$, $\text{CO}_2 = 65.0\%$, $\text{O}_2 = 0.2\%$. Output: **Estimated temperature-derived category: High** with class probabilities.
   - *Error Handling:* Enter negative concentration or sum $>102\%$ to show strict physical validation.
5. **Page E (Research Visualizations):** Examine the 2D $(\text{CH}_4, \text{CO}_2)$ scatter plot and explain why severe smoldering creates a distinct, non-overlapping chemical signature.
6. **Page F (Seven-Day Forecasting Research):** Show the 45.79% forecasting failure, review the 58.9% rapid state reversal in the monitoring time series, and explain why static predictive forecasting is empirically unviable on routine compliance logs.
7. **Page G (Model Evaluation & Limitations):** Review the 95% Clustered Well Bootstrap confidence intervals and target-label provenance audit.

---

## 5. Training the Models

To retrain candidate models from the verified primary dataset and re-serialize artifacts:

```bash
python src/train_model.py
```

This updates:
- `models/three_gas_rf_model.joblib`
- `models/three_gas_lr_model.joblib`
- `models/dummy_classifier.joblib`
- `models/model_metadata.json`

---

## 6. Running Automated Tests

Run the full pytest suite to verify data loading, schema integrity, zero-leakage rules, input validation, and inference schemas:

```bash
pytest tests/ -v
```

---

## 7. Model Evaluation Summary

All metrics recomputed out-of-sample on the chronological holdout ($N=3,611$):

| Model | Feature Set | Accuracy | Macro-F1 | High-Risk Recall | 95% Clustered Well CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | Prior Class | 63.36% | 0.2586 | 0.00% | [61.20%, 65.50%] |
| **Logistic Regression Pipeline** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | 92.72% | 0.9301 | 99.84% | [91.79%, 93.49%] |
| **Random Forest (Production)** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | **96.01%** | **0.9606** | **100.00%** | **[95.42%, 96.62%]** |
| **Random Forest (Spatio-Temporal)** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | **95.74%** | **0.9591** | **100.00%** | **[94.35%, 97.09%]** |
| **7-Day Forecasting RF (Verified Test)** | 10 Feats at $t$ | **35.79%** | **0.2908** | **13.88%** | [33.20%, 38.40%] |
| **7-Day Forecasting RF (Phase 10 Variation)** | 7 Feats at $t$ | **45.79%** | **0.2908** | **8.11%** | [40.12%, 51.46%] |

---

## 8. License & Acknowledgments

- **Primary Data Source:** Missouri Department of Natural Resources (MDNR) public compliance monitoring records for Bridgeton Landfill.
- **Reference Literature:** Sabrin, S. (Rowan Univ, 2018; `Papers/k6.pdf`), Thalhamer (2013), Martin et al. (2013), and 40 CFR §60.753.
- **Academic License:** Distributed solely for academic evaluation and peer review.
