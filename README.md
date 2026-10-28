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
├── app.py                          # Streamlit 8-page interactive research web application
├── requirements.txt                # Versioned Python package dependencies
├── .gitignore                      # Git ignore rules (virtual environments, caches)
├── README.md                       # Master project documentation
│
├── .github/workflows/              # Continuous Integration & Scheduled Update Workflows
│   └── scheduled_ingestion.yml     # Automated weekly/manual GitHub Actions ingestion runner
│
├── src/                            # Modular engineering source code
│   ├── __init__.py
│   ├── ingestion.py                # Automated environmental ingestion pipeline & SQLite manager
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
├── tests/                          # Automated test suite (37 tests across 7 test modules)
│   ├── __init__.py
│   ├── test_ingestion.py           # Ingestion parsing, validation, idempotency, and cache tests
│   ├── test_data_loader.py         # Data loading, row count, and schema tests
│   ├── test_preprocessing_and_splits.py # Zero-leakage and split integrity tests
│   ├── test_validation.py          # Input validation, bounds, and total sum tests
│   ├── test_model_and_inference.py # Model serialization and inference schema tests
│   ├── test_evaluation_metrics.py  # Metric calculations and clustered bootstrap tests
│   └── test_app_startup.py         # App syntax and page registration tests
│
├── data/                           # Segregated persistent storage
│   └── external_monitoring.db      # SQLite database for external observations and audit logs
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
8. **Page H (Data Updates & Monitoring):** Inspect real-time ingestion telemetry, review persistent SQLite observations, examine historical vs. external data segregation, and demonstrate on-demand manual ingestion.

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

Run the full pytest suite (41 tests across 7 test modules) to verify data loading, schema integrity, zero-leakage rules, input validation, inference schemas, and the automated external ingestion pipeline with dual PostgreSQL/SQLite persistence:

```bash
pytest tests/ -v
```

---

## 7. Model Evaluation Summary

All metrics recomputed out-of-sample on the chronological holdout ($N=3,611$):

| Model | Feature Set | Accuracy | Macro-F1 | High-Risk Recall | 95% Clustered Well CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | Prior Class | 59.12% | 0.2477 | 0.00% | [57.49%, 60.83%] |
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

---

## 9. Automated Environmental Ingestion & Persistent Storage Architecture

### 9.1 Overview & Cross-Environment Data Synchronization
The application features a modular ingestion pipeline (`src/ingestion.py`) that periodically retrieves publicly available environmental data without compromising the scientific validity or historical benchmarks of the core machine learning models.

Because GitHub Actions runners and deployed Streamlit applications run in isolated, ephemeral container environments that do not share a local filesystem, the system implements a **dual-backend persistence architecture**:
1. **Managed Cloud PostgreSQL (Production Mode):** When `PERSISTENT_DATABASE_URL` is configured, both the scheduled GitHub Actions workflow and the deployed Streamlit web application connect to the same remote database (e.g., Supabase or Neon). Observations written by the runner persist indefinitely and appear in real time on the website.
2. **Local SQLite Fallback (Development/Offline Mode):** When no remote credentials are provided, or if the remote database is unreachable, the pipeline falls back gracefully to a local SQLite database (`data/external_monitoring.db`). The website marks cloud persistence as `NOT CONFIGURED` or `CONNECTION ERROR`, logs the diagnostic status, and prevents pipeline crashes without ever claiming a false live connection.

```text
[External Data Sources]
  ├── US EPA Envirofacts API (Live Regulatory GHGRP Filings)
  ├── Open-Meteo Weather API (Live Ambient Landfill Meteorology)
  ├── Carbon Mapper / NASA JPL (Airborne/Satellite Methane Plumes)
  └── Synthetic Test Fixture (Offline Unit Testing)
            │
            ▼
[Modular Adapters (src/ingestion.py)]
  ├── Exponential Backoff & Retry Logic (3 attempts)
  ├── Request Timeouts (10-12s)
  └── Timestamp / Coordinate Normalization (UTC ISO-8601)
            │
            ▼
[Scientific Compatibility Gate (validate_classifier_compatibility)]
  ├── Checks: CH4, CO2, O2 present?
  ├── Units: Vol % (0-100%)?
  └── Quarantines Incompatible Data (Flux kg/h, Ambient °C/hPa)
            │
            ▼
[Dual DatabaseManager (src/ingestion.py)]
  ├── If PERSISTENT_DATABASE_URL is set:
  │     └── Managed Cloud PostgreSQL (Supabase / Neon)
  │           ▲                               ▲
  │           │ Writes observations            │ Reads & visualizes
  │     [GitHub Actions Runner]        [Deployed Streamlit App]
  │
  └── If credentials missing / unreachable:
        └── Graceful Fallback to Local SQLite (data/external_monitoring.db)
              └── Streamlit UI shows "Cloud Persistence: NOT CONFIGURED"
```

### 9.2 Verified External Public Sources

| Source Name | Endpoint / Resource | Licensing / Access | Fields Retrieved | Update Cadence | Model Compatibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **US EPA Envirofacts GHGRP** | `https://data.epa.gov/efservice/PUB_DIM_FACILITY/facility_id/1007026/JSON` | Public Domain; Free, No API key required | Facility ID, Reporting Year, Subparts (HH, C), Parent Company, Lat/Lon | Annual regulatory release | **Quarantined:** High-level facility data, not borehole gas % |
| **Open-Meteo Weather** | `https://api.open-meteo.com/v1/forecast?latitude=38.77&longitude=-90.44` | Open-Meteo Open Access; Free non-commercial | Temperature (2m °C), Humidity (%), Pressure (hPa), Wind Speed (km/h) | Hourly / Live | **Quarantined:** Surface weather cannot evaluate downhole reaction |
| **Carbon Mapper Plumes** | `https://api.carbonmapper.org/api/v1/` / `Datasets/Landfill_plume_detections.csv` | CC-BY 4.0; Token or catalog | Plume Candidate ID, Instantaneous Emission Rate (kg/h), Uncertainty | Periodic airborne & satellite overpasses | **Quarantined:** Plume flux (kg/h) is dimensionally incompatible with Vol % |
| **Synthetic Test Fixture** | Internal adapter (`src/ingestion.py`) | Academic test fixture | Simulated 3-gas and plume records | Offline testing | **Marked Synthetic:** `is_synthetic = 1` |

### 9.3 Protecting Scientific Validity
The existing 8 ML classifiers strictly require three input features: `CH4_percent`, `CO2_percent`, and `O2_percent` in **volumetric percentage (Vol %)**.
- **No Direct Satellite Ingestion into Classifiers:** Satellite remote sensing instruments measure atmospheric column enhancement or total emission flux ($\text{kg/h}$). These cannot be converted into subsurface extraction wellhead volume fractions without site-specific transport models and wellfield capture efficiency metrics.
- **No Value Synthesis:** The pipeline never synthesizes missing gases or fabricates dummy values.
- **Physical Boundary:** The system strictly classifies temperature-associated subsurface thermal risk states from empirical gas ratios. It does NOT detect open surface fires or reliably forecast future ignition events.

### 9.4 Step-by-Step Setup: Configuring Cloud Persistence (Supabase / Neon)

To enable automated end-to-end synchronization between scheduled GitHub Actions and the deployed Streamlit website:

#### Step 1: Provision a Free Managed PostgreSQL Database
You can use either [Supabase](https://supabase.com/) or [Neon](https://neon.tech/):
1. **Neon (Recommended for simplicity):**
   - Create a free account at `neon.tech`.
   - Create a new project (e.g., `landfill-gas-monitoring`).
   - Copy the connection string from the dashboard (e.g., `postgresql://user:password@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require`).
2. **Supabase:**
   - Create a free project at `supabase.com`.
   - Go to **Project Settings > Database > Connection String > URI**.
   - Copy the URI with your database password (e.g., `postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`).

#### Step 2: Configure GitHub Repository Secrets (for Scheduled Workflow)
1. Go to your repository on GitHub: **Settings > Secrets and variables > Actions**.
2. Click **New repository secret**.
3. Name: `PERSISTENT_DATABASE_URL`
4. Value: Paste your PostgreSQL connection URI.
5. (Optional) Add `CARBON_MAPPER_API_KEY` if you have an API key for live Carbon Mapper satellite plume queries.

#### Step 3: Configure Streamlit Community Cloud Secrets (for Deployed Website)
1. In your Streamlit Cloud management dashboard, open your app settings.
2. Select **Secrets** (or edit `.streamlit/secrets.toml` locally for local testing):
   ```toml
   PERSISTENT_DATABASE_URL = "postgresql://user:password@host:port/dbname?sslmode=require"
   ```
3. Save the secrets. The Streamlit app will automatically reconnect to the shared PostgreSQL store without restarting the container.

#### Step 4: Verify Synchronization
1. Run the GitHub Actions workflow manually via **Actions > Scheduled Environmental Data Ingestion > Run workflow**.
2. Inspect the pre-flight check log: it will output `Cloud Persistence: CONFIGURED (Managed PostgreSQL)`.
3. Open your deployed Streamlit website and navigate to **Page H (Data Updates & Monitoring)**:
   - The persistence banner will display: `Cloud Persistence ACTIVE (Managed PostgreSQL): Synchronized to shared database at postgresql://user:****@host:port/dbname`.
   - The stored records, latest observation timestamp, and last ingestion sync time will update in real time.

### 9.5 Local Execution & Manual Ingestion
To manually trigger an ingestion run from the terminal:

```bash
# Ingest live regulatory records from US EPA Envirofacts:
python src/ingestion.py --source US_EPA_Envirofacts_GHGRP --max-records 20

# Ingest live ambient weather at Bridgeton Landfill:
python src/ingestion.py --source Open_Meteo_Ambient_Weather

# Ingest satellite plume detections:
python src/ingestion.py --source Carbon_Mapper_Plume_Catalog --max-records 10

# Run offline synthetic test fixture:
python src/ingestion.py --source Synthetic_Test_Fixture

# Target a specific PostgreSQL database:
python src/ingestion.py --source US_EPA_Envirofacts_GHGRP --db-url "postgresql://user:pass@host:5432/db"
```

### 9.6 Ingestion Pipeline Status

| Component | Status | Verification & Notes |
| :--- | :--- | :--- |
| **EPA Envirofacts Connector** | **Implemented & Tested** | Verified against live EPA REST endpoint (Facility ID 1007026, 14 years ingested). |
| **Open-Meteo Weather Connector** | **Implemented & Tested** | Verified against live Open-Meteo API for Bridgeton Landfill coordinates. |
| **Carbon Mapper Catalog Adapter** | **Implemented & Tested** | Verified with confirmed plume catalog (1,714 records) & API token support. |
| **Synthetic Mock Adapter** | **Mocked for Testing Only** | Explicitly marked `is_synthetic = 1` and `data_quality_status = 'TEST_FIXTURE'`. |
| **Dimensional Unit Validator** | **Implemented & Tested** | Blocks non-Vol % data (kg/h, °C, hPa) from 3-gas classifiers with explicit reasons. |
| **Managed PostgreSQL Adapter** | **Implemented & Tested** | Dual backend with automatic migration, schema creation, and idempotent deduplication (`ON CONFLICT`). |
| **Local SQLite Fallback** | **Implemented & Tested** | Automatically falls back to `data/external_monitoring.db` if credentials absent or network fails. |
| **Password Masking** | **Implemented & Tested** | Credentials masked (`:****@`) in logs, CLI output, and Streamlit UI. |
| **Streamlit Monitoring Page** | **Implemented & Tested** | Page H active with live persistence indicator, stale-data warnings, telemetry, and manual refresh. |
| **Cache Invalidation Engine** | **Implemented & Tested** | `get_cached_external_observations.clear()` reloads data; model caches remain intact. |
| **Scheduled GitHub Actions** | **Implemented & Tested** | Workflow in `.github/workflows/scheduled_ingestion.yml` with pre-flight check & secret mapping. |
| **Cloud Persistence Configuration** | **Adapter Ready; Awaiting Credentials** | Requires user to create Supabase/Neon instance and supply `PERSISTENT_DATABASE_URL`. |

