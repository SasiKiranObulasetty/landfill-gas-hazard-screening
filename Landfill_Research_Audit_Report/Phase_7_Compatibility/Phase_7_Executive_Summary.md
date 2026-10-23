# Phase 7 Executive Summary: Scientific Dataset Compatibility Audit

**Project Title:** AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills  
**Supervising Context:** Final-Year Undergraduate Engineering Research Project  
**Author:** Senior Environmental Data Scientist & Audit Agent  
**Deliverable Directory:** `Landfill_Research_Audit_Report/Phase_7_Compatibility/`  
**Completion Date:** October 2026  

---

## 1. What Can Be Modelled NOW

Based on strict empirical verification of schemas, coordinate reference systems, measurement units, and ground-truth labels, the following two machine learning models can be built **immediately with zero additional data collection**:

### Pipeline 1: Subsurface Gas-Thermal Reaction Hazard Classifier (Primary Core)
- **Primary Source:** `Datasets/datasets.zip -> k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx` (18,469 rows) with cross-validation on `ETLF Gas SciHub Final.xlsx` (43,534 rows).
- **Target Variable:** `Risk_Category` ($	ext{Low/Normal}$ vs. $	ext{Elevated Temperature}$ vs. $	ext{Subsurface Reaction / Fire}$).
- **Key Features:** $CH_4$ (% vol), $CO_2$ (% vol), $O_2$ (% vol), Residual $N_2$ (% vol), Temperature (°C), $CH_4/CO_2$ ratio, and derived Thermal Gas Index.
- **Scientific Value:** Solves the core engineering problem of early subsurface smoldering fire detection before surface breakout occurs. Fully defensible, peer-reviewed methodology.

### Pipeline 2: Landfill Methane Super-Emitter Hotspot Predictor (Secondary Core)
- **Primary Source:** Direct join of `Datasets/Landfill_plume_detections.csv` (1,262 valid detections) with `Datasets/landfilllmopdata.xlsx` (2,641 landfills) on `GHGRP ID`.
- **Target Variable:** Plume emission rate flux ($	ext{kg/h}$) and binary super-emitter threshold ($>100	ext{ kg/h}$).
- **Key Features:** Waste-in-place tonnage, operational status (Open/Closed), gas collection infrastructure, flare capacity, surface area.
- **Scientific Value:** Identifies which landfills are prone to high-emission fugitive methane plumes using facility attributes. Evaluated via Group-K-Fold by facility to guarantee zero spatial leakage.

---

## 2. What CANNOT Yet Be Modelled (and Why)

### 1. End-to-End Multi-Source Short-Term Fire Forecasting
- **The Issue:** Attempting to build a single tabular model that joins the 2013-2019 US subsurface gas measurements with 2025 Indian MODIS thermal detections and 2022 New York mobile survey transects is **scientifically invalid**.
- **Reason:** The datasets are completely disjoint across time ($2013-2019$ vs. $2021$ vs. $2025$) and space (Missouri/USA vs. Western NY/USA vs. Bangladesh vs. India). Merging them into a single row would produce fabricated, spurious correlations.

### 2. Direct Satellite Thermal Dumpsite Fire Detection (Without GIS Buffering)
- **The Issue:** `modis_2025_India (1).csv` cannot be used raw as a "landfill fire" training set.
- **Reason:** A MODIS pixel covers $1	ext{ km} 	imes 1	ext{ km}$ to $2	ext{ km} 	imes 4.8	ext{ km}$ ($1	ext{ km}^2$ to $9.6	ext{ km}^2$). Across India's 94,917 detections, over $98\%$ represent agricultural crop residue stubble burning, forest fires, or industrial heat. It becomes usable **only after spatial intersection** with the 1,137 landfill coordinates in `data/data/ClimateTrace/waste/waste_geometries.gpkg`.

### 3. Open Dumpsite Surface Flame Prediction from In-Situ Wellhead Data
- **The Issue:** Wellhead telemetry from closed US landfills cannot be directly extrapolated to predict spontaneous surface open flaming at unmanaged open dumps (e.g. Ghazipur or Deonar in India) without operational domain adaptation.
- **Reason:** Managed landfills contain hermetic plastic liners and vacuum extraction wells; open dumps burn aerobically on exposed surface slopes.

---

## 3. Concrete Next Steps for the Final-Year Project

To achieve the highest marks from the faculty supervisor and examination committee, execute the project in three sequential, clearly documented steps:

### Step 1: Execute Pathway 1 (Subsurface Gas-Thermal Risk Classifier)
1. Write a Python script to extract and preprocess `k6_bridgeton` (18,469 rows).
2. Establish a strict **Temporal Train/Test Split** (Train on 2013-2016, Test on 2017-2019).
3. Train baseline classifiers: Logistic Regression, Random Forest, and Gradient Boosted Trees (XGBoost/LightGBM).
4. Evaluate with Precision-Recall AUC, F1-score, and Confusion Matrix.
5. Perform out-of-distribution validation against `ETLF Gas SciHub Final.xlsx` alert thresholds and the 60 Bubanj Landfill baseline rows.

### Step 2: Execute Pathway 2 (Remote Sensing Methane Hotspot Model)
1. Join `Landfill_plume_detections.csv` and `landfilllmopdata.xlsx` on `GHGRP ID`.
2. Clean trailing blanks and impute missing operational flags.
3. Train a gradient boosted regressor with **Group-K-Fold by Landfill ID**.
4. Evaluate feature importances (e.g. SHAP values) to prove which physical landfill features drive fugitive methane plumes.

### Step 3: Spatial Dumpsite Screening (GIS Enrichment)
1. Run a spatial distance query between `modis_2025_India (1).csv` (94,917 rows) and the 1,137 Indian solid waste sites in `waste_geometries.gpkg`.
2. Extract the subset of thermal detections occurring within a $2	ext{ km}$ buffer radius of verified dumpsites.
3. Generate a ranked spatial map / risk table of thermal hotspot frequency across Indian municipal dumpsites for 2025.

---

## 4. Phase 7 Deliverables Summary

All 6 required Phase 7 deliverables have been compiled and verified in [Landfill_Research_Audit_Report/Phase_7_Compatibility/](file:///c:/Users/obula/OneDrive/Desktop/4th_year_project/Landfill_Research_Audit_Report/Phase_7_Compatibility):

1. **`Dataset_Schema_Audit.csv`**: Detailed 16-attribute schema audit for every candidate dataset (rows, cols, units, CRS, temporal resolution, quality issues, genuine labels).
2. **`Dataset_Compatibility_Matrix.csv`**: Pairwise scientific compatibility evaluation across 14 dataset combinations with exact join keys and risk justifications.
3. **`Data_Quality_Findings.md`**: Comprehensive data quality, unit standardization, missing value, and ground-truth vs. proxy integrity report.
4. **`Feasible_Modelling_Tasks.md`**: Rigorous evaluation of Tasks A through E with explicit data leakage and sample size assessments.
5. **`Recommended_Minimal_Dataset.md`**: Full experimental specification for the two defensible modeling pathways.
6. **`Phase_7_Executive_Summary.md`**: High-level supervisor briefing on what can be modeled now, what cannot, and the next concrete step.
