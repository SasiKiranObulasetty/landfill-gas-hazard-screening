# Executive Dataset Decision & Research Feasibility Assessment

**Project Title:** AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills  
**Audit Status:** Complete Read-Only Audit  
**Author / Investigator:** Environmental Data Science & Machine Learning Research Team  
**Date:** October 2026  

---

## 1. Executive Summary

This audit rigorously evaluated **828 total inventoried files**, comprising **22 distinct dataset files/sub-datasets** and **39 research paper PDFs** (representing **32 unique scientific research papers** and 7 duplicate/draft copies). 

The investigation confirmed that the repository contains exceptionally high-quality empirical data across three distinct domains:
1. **Spaceborne & Airborne Methane Plume Remote Sensing:** High-resolution point-source plumes from GHGSat (~25m, 1,519 plumes) and Carbon Mapper (~5m, 2,527 plumes), alongside hyperspectral satellite super-emitter catalogs (EMIT, EnMAP, PRISMA).
2. **Ground-Truth Subsurface Elevated-Temperature & Smouldering Reaction Data:** Over **38,000 in-situ wellhead observations** from Bridgeton Landfill (Paper `k6`, 18,469 rows) and Ohio Elevated Temperature Landfills (`ETLF Gas SciHub Final.xlsx`, >20,000 rows), tracking multi-gas concentrations ($\text{CH}_4$, $\text{CO}_2$, $\text{O}_2$, $\text{CO}$, $\text{H}_2$), internal temperatures up to 85°C+, and explicit risk/alert classifications.
3. **Multi-Source Thermal Remote Sensing & Meteorology:** 24 consecutive years of synchronized monthly Land Surface Temperature (LST), NDVI, NDBI, humidity, and waste mass from Rajbandh Landfill (`k1`), >312,000 rows of continuous mobile laboratory transects in New York, and full-year 2025 nationwide MODIS active fire detections across India.

### Core Strategic Verdict:
* **Methane Hotspot Detection:** **100% Feasible Immediately** using the existing satellite and airborne datasets.
* **Fire-Risk Assessment & Hazard Scoring:** **100% Feasible Immediately** using the Bridgeton (`k6`), Ohio ETLF, and Rajbandh (`k1`) datasets.
* **Future Fire-Risk Forecasting (Forward Temporal Prediction):** **Not Defensible with Current Files Alone**. True future prediction requires synchronized pre-fire sensor streams and confirmed fire ignition timestamps.
* **Recommended Framework:** Build an **Interpretable Multi-Source Landfill Risk-Screening and Hotspot Prioritization Engine** that integrates satellite plume detection, surface thermal anomalies, meteorological vulnerability, and ground-truth subsurface reaction scoring.

---

## 2. Final Dataset Selection & Categorization

The available data files have been categorized into a lean, scientifically defensible collection:

### A. Final Core Datasets (Retain for Implementation)
1. **`k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx` (in `Datasets/datasets.zip`):**
   * *Role:* Supervised ground-truth training dataset for internal thermal-gas risk classification. Contains 18,469 labeled rows classifying wells into Low, Medium, and High Risk of subsurface smouldering reaction based on $\text{CH}_4/\text{CO}_2$ ratio, $\text{O}_2$, residual $\text{N}_2$, and internal temperature.
2. **`ETLF Gas SciHub Final.xlsx`:**
   * *Role:* Multi-site cross-validation dataset. Provides >20,000 empirical wellhead measurements across 5 distinct North American landfills (ETLF A–E) measuring internal temperatures (up to 85°C+), $\text{CH}_4$, $\text{CO}_2$, $\text{CO}$ (ppmv), $\text{H}_2$, and alert levels.
3. **`GHGSat_detected_plumes.csv`:**
   * *Role:* High-resolution spaceborne methane plume ground truth. Contains 1,519 quantified point-source plumes ($Q$ in t/h, coordinates, wind speed, IME) across 151 global landfills.
4. **`k1_khulna_rajbandh_full_monthly_dataset_2000_2023.xlsx` (in `Datasets/datasets.zip`):**
   * *Role:* Regional South Asian multi-source data fusion anchor. 289 consecutive months linking satellite Land Surface Temperature (LST), vegetation index (NDVI), built-up index (NDBI), humidity, wind, and landfill waste tonnage.
5. **`landfills_hyper_obs.xlsx` (in `data/data/`):**
   * *Role:* Geospatial facility registry for global super-emitting landfills (82 sites across 33 countries), providing exact coordinates for Indian mega-dumps (Ghazipur, Bhalswa, Deonar) and satellite-quantified emission baselines.
6. **`modis_2025_India (1).csv`:**
   * *Role:* Surface thermal anomaly proxy for India. 94,917 active fire/thermal detections across 2025; spatially filtered via 1.5 km buffer around known Indian landfill coordinates to quantify seasonal thermal hotspot recurrence.

### B. Supporting Datasets (Retain for Validation & Context)
7. **`GHGSat_landfill_survey_Dataset.xlsx`:** Facility-level benchmark comparing satellite flux against LandGEM FOD and California CEA regulatory models across 60 North American landfills.
8. **`carbonmapper_ch4_plumelist_2020_2021.xls`:** 2,527 airborne hyperspectral plumes (~5m resolution) for ultra-high-resolution spatial plume boundary validation.
9. **`Landfill_plume_detections.csv`:** Connects Carbon Mapper plume hotspots to US EPA GHGRP facility IDs (after removing 451 duplicate rows).
10. **`landfilllmopdata.xlsx`:** Authoritative engineering database of 2,642 US MSW landfills (waste-in-place, gas collection status, depth, acreage, coordinates).
11. **`waste_geometries.gpkg` (in `data/data/ClimateTrace/`):** Global spatial database of 62,662 solid waste site points used as a spatial mask to prevent false-positive fire associations.
12. **`DATASETS (1).zip -> Data.zip` (14 CSVs):** >312,000 rows of continuous mobile laboratory transects across 5 New York landfills, validating micro-meteorological methane dispersion.
13. **`k5_bubanj_full_field_measurements_60_rows.xlsx` (in `Datasets/datasets.zip`):** Empirical field evidence linking ambient temperature, well temperature, and carbon monoxide (CO) smouldering thresholds.
14. **`655880d7fc235e0aa45f8b8cbbf140f5.zip`:** ERA5 hourly weather reanalysis NetCDF pipeline (2,952 hours) demonstrating automated temperature and wind extraction.

### C. Optional & Excluded Datasets
* **`k4_complete_ffta_study_data.xlsx` (Optional):** Fuzzy fault tree engineering probability tables. Excellent for qualitative risk weighting, but not suitable for statistical machine learning training.
* **`monthly_results.csv`, `pressure_dependence_results.csv`, `temperature_dependence_results.csv` (Optional):** Binned summary regression tables from flux tower studies. Retained as physical reference priors.
* **`Hyper_plume_cases/*.nc` (Optional):** Pixel-level hyperspectral plume NetCDFs (Kabd, Norte III, Tecnosilicatos). Useful for computer-vision segmentation demos (PRISMethaNet style), but optional for tabular risk modeling.
* **`Landfill_plume_detections (2).csv` (EXCLUDED - Redundant):** Exact 100% duplicate of `Landfill_plume_detections.csv`.
* **`carbonmapper_ch4_plumelist_2020_2021 (1).xls` (EXCLUDED - Redundant):** Exact 100% duplicate of `carbonmapper_ch4_plumelist_2020_2021.xls`.
* **`p3.pdf`, `p4.pdf`, `Global Identification...Using.pdf`, `Thermal hotspot detection in landfills.pdf` (EXCLUDED - Duplicates):** Duplicate research papers identical to full-title PDFs.
* **`The toxic impacts of microplastics (MPs)...bivalve species.pdf` (EXCLUDED - Irrelevant):** Marine clam toxicology paper with zero relevance to landfills.

---

## 3. Feasibility of Fire-Risk Prediction: Critical Scientific Evaluation

The research prompt mandates assessing whether the available data supports a genuine predictive model. We must distinguish four distinct scientific tasks:

```
Task 1: Methane Hotspot Detection
[STATUS: FULLY SUPPORTED]
- Data: GHGSat (1,519 plumes), Carbon Mapper (2,527 plumes), Hyper-Landfill (82 sites).
- Capability: Spatial localization of massive emission plumes, flux rate quantification (kg/h), and identification of persistent emission zones.

Task 2: Historical Thermal Anomaly & Fire Screening
[STATUS: FULLY SUPPORTED]
- Data: MODIS 2025 India (94,917 detections), Sarimukti study (Paper 3), Brahmapuram fire (Paper 10).
- Capability: Spatial intersection of thermal anomalies with landfill boundaries; seasonal frequency analysis; burn severity mapping.

Task 3: Multi-Factor Subsurface Fire-Risk Assessment
[STATUS: FULLY SUPPORTED]
- Data: Bridgeton Landfill (18,469 rows with RSET risk categories), Ohio ETLF (>20k rows with alert levels).
- Capability: Training supervised ML models (Random Forest, XGBoost, Neural Networks) to classify landfill risk levels (Low, Medium, High) based on gas ratios (CH4/CO2), O2 intrusion, CO ppm, and internal temperatures.

Task 4: Future Fire-Risk Prediction (Forward Forecasting)
[STATUS: NOT SUPPORTED BY CURRENT FILES ALONE]
- Missing: Synchronized continuous time series preceding verified ignition timestamps across multiple sites; confirmed non-fire control intervals.
- Scientific Reality: Predicting that a fire will break out at Landfill X on day T+7 requires temporal feature lags that current snapshot datasets do not possess.
```

> [!IMPORTANT]
> **Academic Integrity Warning:** Claiming that we have trained a "validated future fire prediction AI" using satellite snapshots and disconnected 2025 MODIS fire points is scientifically indefensible and will be dismantled in a rigorous thesis defense. The honest, defensible contribution is **Multi-Source Hazard Screening and Fire-Risk Assessment**.

---

## 4. Research Novelty & Literature Gap Supported by the Papers

Our exhaustive review of the 32 unique research papers revealed three profound structural gaps in the international literature:

1. **The Remote Sensing vs Ground Monitoring Silo:**
   * Studies either focus on spaceborne methane detection without considering internal thermal/fire conditions (Maasakkers 2025, Cusworth 2025, PRISMethaNet 2024), OR they focus exclusively on internal well chemistry without spatial remote sensing (Bridgeton `k6`, Ohio ETLF).
   * **Our Novelty:** Formulating a unified multi-tier risk screening architecture that bridges satellite surface plume observations, downscaled land surface temperatures (LST), and in-situ gas well risk thresholds.
2. **The Disconnect Between Regulatory Models and Real-World Emissions:**
   * Nature (2025) and ES&T (2022) proved that regulatory inventory models (EPA LandGEM / FOD) show virtually zero correlation with satellite-observed methane emissions.
   * **Our Novelty:** Incorporating empirical satellite plume detections and environmental covariates (temperature, barometric pumping) to dynamically adjust facility-level risk scores.
3. **Absence of Explainable Multi-Hazard Frameworks for Developing Nations:**
   * Existing fire risk frameworks rely on expensive in-situ sensor networks unavailable at unmanaged open dumpsites in India and South Asia.
   * **Our Novelty:** Designing an uncertainty-aware, interpretable screening framework (leveraging SHAP attribution and Fuzzy Fault Tree cut-sets) specifically adapted to data-constrained waste sites.

---

## 5. What We Can Honestly Claim to Mam (Faculty Supervisor)

Here is the exact, academically rigorous wording recommended for your presentation to your faculty supervisor:

> **Supervisor Presentation Script:**
> *"Respected Mam, after conducting a rigorous, read-only audit of all 828 project files, 22 datasets, and 32 recent research papers, we have established a strong, scientifically defensible path forward:*
>
> 1. *We have confirmed that we possess gold-standard ground-truth datasets for landfill gas hazards: over 38,000 in-situ monitoring rows from Bridgeton Landfill and Ohio Elevated Temperature Landfills, which provide exact chemical measurements ($\text{CH}_4$, $\text{CO}_2$, $\text{O}_2$, $\text{CO}$, and temperatures up to 85°C) and verified fire-risk classifications.*
> 2. *We also possess global satellite and airborne methane plume catalogs (GHGSat and Carbon Mapper) covering thousands of high-resolution point-source detections, alongside nationwide 2025 MODIS active fire detections across India.*
> 3. *Based on recent literature (Nature 2025, Waste Management 2026), we must make an honest scientific distinction: the data fully supports **Methane Hotspot Detection** and **Multi-Factor Fire-Risk Assessment**, but does not support claiming an unvalidated 'future fire prediction' model due to satellite revisit gaps and absence of continuous pre-fire sensor streams.*
> 4. *Therefore, our defensible research novelty is an **Interpretable Multi-Source Landfill Hazard-Screening Framework**. We will train supervised machine learning models on our rich ground-truth wellhead datasets to predict subsurface reaction risk, and fuse these with satellite-derived thermal indicators and meteorological drivers to rank and prioritize landfill hotspots.*
> 5. *All core datasets are already on disk and verified. We can begin immediate implementation without requiring any new downloads."*

---

## 6. Final Decision Summary (Addressing the 7 Core Questions)

1. **Are the existing datasets sufficient to start implementation?**  
   **YES (Confidence: 100%)**. The existing datasets provide over 38,000 ground-truth wellhead rows, >4,000 satellite/airborne methane plumes, 289 months of continuous LST-weather observations, and 94,917 Indian thermal detections. All data loaders and schemas are audited and ready.

2. **Are they sufficient to train and evaluate a credible fire-risk prediction model?**  
   **YES for Fire-Risk Assessment / Hazard Classification (Confidence: 95%)** using Bridgeton (`k6`) and Ohio ETLF data.  
   **NO for Forward Temporal Fire Outbreak Prediction (Confidence: 90%)** due to lack of synchronized pre-fire streaming sensors.

3. **Which datasets should be retained for the final project?**  
   **Retain 6 Core Datasets:** `k6` Bridgeton, `ETLF Gas SciHub`, `GHGSat_detected_plumes`, `k1` Khulna Rajbandh, `landfills_hyper_obs`, and `modis_2025_India`. Retain the 8 supporting datasets for context and validation.

4. **Which datasets are redundant or unnecessary?**  
   `Landfill_plume_detections (2).csv` (100% duplicate), `carbonmapper_ch4_plumelist_2020_2021 (1).xls` (100% duplicate), duplicate paper PDFs (`p3`, `p4`, `Global Identification...Using.pdf`, `Thermal hotspot detection...pdf`), and the non-landfill marine toxicology paper (`The toxic impacts of microplastics...pdf`).

5. **What, if anything, must still be downloaded?**  
   **NOTHING is required to begin implementation (Confidence: 95%)**.  
   *Optional Future Enhancement:* If the supervisor insists on forward fire forecasting for India, download **VIIRS 375m active fire data (2021–2024)** from NASA FIRMS for 5 Indian landfills to align temporally with satellite plume catalogs.

6. **What is the strongest defensible novelty supported by the reviewed papers?**  
   **A Multi-Tier Risk-Screening and Hotspot Prioritization Engine** that integrates spaceborne methane plumes, downscaled satellite land surface temperature (LST), and in-situ multi-gas reaction thresholds, evaluated with SHAP explainability.

7. **What are the three most important next steps?**  
   * **Step 1:** Train baseline supervised classifiers (Random Forest, XGBoost, MLP) on the Bridgeton `k6` dataset (18,469 rows) to predict RSET subsurface fire-risk categories from gas concentrations and ratios.  
   * **Step 2:** Spatial buffer extraction of MODIS 2025 thermal anomalies over Indian landfills (Ghazipur, Bhalswa, Deonar) from `landfills_hyper_obs.xlsx` to analyze seasonal heat recurrence.  
   * **Step 3:** Implement an interpretable composite risk-scoring formula that ranks landfill sectors by combining satellite plume intensity, surface thermal anomalies, and meteorological risk factors.
