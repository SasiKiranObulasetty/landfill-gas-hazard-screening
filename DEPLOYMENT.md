# Deployment Guide: Landfill Hazardous Gas Screening AI Portal

This guide provides step-by-step instructions for deploying the Streamlit research application publicly to **Streamlit Community Cloud** using **GitHub**.

---

## 1. Deployment Architecture & File Inventory

The application is completely self-contained and ready for cloud hosting without external databases or paid API keys:

- **Entrypoint:** `app.py`
- **Dependency Manifest:** `requirements.txt`
- **Core Engine & Utilities:** `src/` (`data_loader.py`, `preprocessing.py`, `predict.py`, `evaluate.py`, `train_model.py`, `visualizations.py`, `reporting.py`)
- **Primary Research Dataset:** `Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx` (~1.0 MB)
- **Trained Model Artifacts:** `models/` (`three_gas_rf_model.joblib`, `three_gas_lr_model.joblib`, `dummy_classifier.joblib`, `model_metadata.json`) (~1.55 MB)
- **Total Deployable Repository Size:** **< 20 MB** (well below GitHub's 100 MB file limit and Streamlit Cloud's 1 GB repository limit).
- **Excluded Heavy Assets:** The 4 GB satellite imagery exploration directory (`data/`) is ignored via `.gitignore` to prevent repository bloat and deployment failures.

---

## 2. Prerequisites

1. A [GitHub](https://github.com) account.
2. A [Streamlit Community Cloud](https://share.streamlit.io) account (connected to your GitHub account).
3. Git installed on your local computer.

---

## 3. Step-by-Step GitHub Setup

Open a terminal (PowerShell or Bash) in the project directory: `C:\Users\obula\OneDrive\Desktop\4th_year_project`.

### Step 3.1: Initialize Git and Check Status
```bash
git init
git status
```
> **Verification Check:** Ensure that `.venv/`, `data/`, and cache directories are marked as untracked/ignored. `Confirmed_Minimal_Datasets/`, `models/`, `src/`, `reports/`, `app.py`, and `requirements.txt` must be tracked.

### Step 3.2: Stage and Commit the Code
```bash
git add .
git commit -m "feat: prepare landfill gas hazard screening app for public deployment"
```

### Step 3.3: Create a Remote Repository on GitHub
1. Log in to [GitHub](https://github.com).
2. Click the **"+"** icon in the top right corner and select **New repository**.
3. Set the repository name (e.g., `landfill-gas-hazard-screening` or `4th_year_project`).
4. Select **Public** (or **Private**; Streamlit Community Cloud supports both).
5. Do **not** initialize with a README, .gitignore, or license (these already exist locally).
6. Click **Create repository**.

### Step 3.4: Link Remote and Push
Copy your repository URL from GitHub and run:
```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git branch -M main
git push -u origin main
```

---

## 4. Deploying on Streamlit Community Cloud

### Step 4.1: Create New App
1. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** (or **New app**).
3. Choose **"I already have an app"**.

### Step 4.2: Configure Deployment Settings
Fill in the form fields:
- **Repository:** `<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`
- **Branch:** `main`
- **Main file path:** `app.py`
- **App URL (optional):** Enter a custom subdomain (e.g., `landfill-gas-screening.streamlit.app`).

### Step 4.3: Advanced Settings (Recommended)
1. Click **Advanced settings**.
2. **Python version:** Select **`3.11`** or **`3.12`**.
3. **Secrets:** Leave empty. *(The application does not require any API keys, credentials, or database connection strings).*
4. Click **Save**.

### Step 4.4: Launch Deployment
Click **Deploy!**.

Streamlit will provision the container, install the dependencies from `requirements.txt`, load the cached dataset, and launch the portal. The deployment process typically takes 1–2 minutes.

---

## 5. Post-Deployment Verification Checklist

Once the deployment finishes and your public URL opens:

| Page | Verification Test | Expected Behavior |
| :--- | :--- | :--- |
| **Page A — Overview** | Verify top metric cards & disclaimers | Shows 18,469 observations, 150 wells, 3 gases, 96.0% holdout accuracy. |
| **Page B — Dataset Explorer** | Test filter sliders & table preview | Displays records, allows wellhead filtering, enables CSV export. |
| **Page C — Classification** | Switch between Random Forest and Logistic Regression | Confirms confusion matrix and 95% clustered bootstrap confidence intervals. |
| **Page D — Interactive Screening** | Input test values ($52\%\,\text{CH}_4, 40\%\,\text{CO}_2, 0.5\%\,\text{O}_2$) | Outputs "Low/Normal" risk category with model confidence. |
| **Page E — Visualizations** | Check Plotly charts | Interactive scatter plot and gas distributions render smoothly. |
| **Page F — Forecasting** | Check 7-day research findings | Displays benchmark comparison table and transition matrix. |
| **Page G — Evaluation** | Verify academic literature table | Displays comparative table of reviewed research papers. |

---

## 6. Built-in Fault Tolerance & Error Handling

- **Automatic Model Retraining Fallback:** If Python or scikit-learn version differences in the Linux container prevent loading the Windows-trained `.joblib` files, `src/predict.py` catches the unpickling exception and automatically retrains and saves compatible model artifacts on demand in ~2 seconds.
- **Read-Only Dataset Safety:** `src/data_loader.py` strictly accesses the source dataset in read-only mode and caches the DataFrame in memory (`@st.cache_data`).
- **Memory Footprint:** The application consumes approximately 140 MB of RAM, well within Streamlit Cloud's 1 GB free-tier threshold.

---

## 7. Sharing with Your Supervisor

Once verified, copy the public URL (e.g. `https://<your-subdomain>.streamlit.app`) and share it with your supervisor.
You can also share:
- Link to the GitHub repository.
- Recompiled final research report: `reports/Final_Research_Report.pdf`.
