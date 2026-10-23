# Comprehensive Research Audit and Final Dataset Recommendation Report

**Submitted for Academic Review & Faculty Supervisor Approval**  
**Project Title:** AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills  
**Academic Year:** 2026–2027 (Final Year B.Tech / M.Tech Research Project)  
**Audit Protocol:** Complete Non-Destructive, Read-Only Audit  
**Location of Audit Outputs:** `c:\Users\obula\OneDrive\Desktop\4th_year_project\Landfill_Research_Audit_Report`  

---

## 1. Project Context and Research Objectives

Solid waste landfills represent complex biogeochemical reactors where anaerobic decomposition generates vast quantities of methane ($\text{CH}_4$) and carbon dioxide ($\text{CO}_2$). In unmanaged open dumps and semi-aerobic disposal cells, localized air intrusion, biological heating, and chemical oxidation can trigger catastrophic subsurface smouldering reactions and surface fires. 

The primary objective of this project is to investigate whether multi-source observational data—combining satellite remote sensing, airborne imaging spectrometry, municipal landfill operational databases, meteorological reanalysis, and in-situ gas well telemetry—can be fused into a scientifically defensible Artificial Intelligence (AI) framework for:
1. **Detecting hazardous methane and gas-related hotspots** in solid waste landfills.
2. **Analyzing empirical relationships** between methane emissions, thermal/fire anomalies, weather conditions, and internal landfill temperatures.
3. **Assessing multi-factor fire and elevated-temperature risk** using machine learning.
4. **Establishing a defensible research novelty** through multi-source data fusion.

This comprehensive audit was commissioned to rigorously inspect all available local research materials, verify data quality, eliminate duplicate or unviable sources, assess compatibility, and formulate an evidence-based recommendation for project execution.

---

## 2. Summary of Inventoried Materials

The read-only audit examined every file across all primary directories, nested subfolders, and internal archive containers:

* **Total Files Inventoried:** **828 files** (all accounted for in `File_Inventory.csv`).
* **Tabular & Scientific Datasets:** **413 dataset files** (including CSVs, multi-sheet Excel workbooks, NetCDF multi-dimensional climate grids, and GeoPackage vector databases).
* **Research Papers & Technical Reports:** **41 PDF files**, comprising **32 unique scientific research papers**, 7 duplicate/draft copies, and 2 dataset documentation manuals.
* **Geospatial & Visual Assets:** **234 satellite plume masks / RGB figures**, 6 animation GIFs, and 4 interactive visualization HTML maps.
* **Operating System Artifacts:** **60 macOS metadata files** (`.DS_Store` and `._*` resource forks from prior unzipping on Apple systems), cleanly segregated.

All 828 files were inspected without altering the original timestamps, names, or directory structures.

---

## 3. Final Dataset Recommendation & Roles

To ensure high scientific rigor and avoid data bloat, we recommend retaining a streamlined collection organized into Core and Supporting datasets:

### Table 1: Final Recommended Dataset Collection

| Dataset Name | Primary File Name | Observation Period | Sample Size / Dimensions | Key Variables Available | Project Role & Research Contribution | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Bridgeton Subsurface Fire Monitoring** | `k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx` | 2013–2018 | 18,469 rows, 10 columns | Well ID, $\text{CH}_4\%$, $\text{CO}_2\%$, $\text{O}_2\%$, $\text{N}_2\%$, Temp (°C/°F), $\text{CH}_4/\text{CO}_2$ ratio, **`Risk_Category`** | **Core Training Dataset:** Supervised ground-truth for classifying subsurface elevated temperature and fire risk. | **A (Core)** |
| **Elevated Temperature Landfills (ETLF)** | `ETLF Gas SciHub Final.xlsx` | 2012–2019 | >20,000 rows across 15 sheets | 5 Landfills (A–E), $\text{CH}_4\%$, $\text{CO}_2\%$, $\text{CO}$ (ppmv), $\text{H}_2\%$, Temp (°C up to 85°C+), **Alert Levels** | **Core Validation Dataset:** Independent multi-site cross-validation of thermal-gas reaction chemistry and smouldering thresholds. | **A (Core)** |
| **GHGSat High-Resolution Satellite Plumes** | `GHGSat_detected_plumes.csv` | 2021–2022 | 1,519 plume rows, 21 columns | Plume coordinates, Emission rate ($Q$ in t/h), Uncertainty, Wind speed, Integrated Methane Enhancement (kg) | **Core Hotspot Dataset:** Spaceborne satellite ground truth for facility-scale methane plume detection and quantification. | **A (Core)** |
| **Rajbandh Multi-Source Time-Series** | `k1_khulna_rajbandh_full_monthly_dataset_2000_2023.xlsx` | 2000–2023 (24 years) | 289 monthly rows, 10 columns | Land Surface Temp (LST °C), NDVI, NDBI, Humidity, Wind speed, Waste tonnage, Methane generation | **Core Regional Fusion Dataset:** South Asian tropical landfill evidence proving multi-source fusion of satellite LST, weather, and waste mass. | **A (Core)** |
| **Global Hyperspectral Landfill Super-Emitters** | `landfills_hyper_obs.xlsx` | 2022–2024 | 82 facilities across 33 countries | Coordinates, Facility names (Ghazipur, Bhalswa, Deonar, Norte III), EMIT/EnMAP/PRISMA detections, Climate TRACE | **Core Geospatial Anchor:** Connects global super-emitter coordinates to satellite observations, anchoring Indian case studies. | **A (Core)** |
| **MODIS Active Fire / Thermal Anomalies India** | `modis_2025_India (1).csv` | Full Year 2025 | 94,917 fire rows, 15 columns | Latitude, Longitude, Brightness temp (K), Fire Radiative Power (MW), Confidence (%), Acquisition date/time | **Supporting Thermal Dataset:** Surface thermal anomaly screening over Indian landfills via 1.5 km spatial buffer filtering. | **B (Supporting)** |
| **EPA LMOP Landfill Database** | `landfilllmopdata.xlsx` | Historical to 2024 | 2,642 US landfills, 42 columns | Exact coordinates, Waste in place, Depth, Liner type, Methane generated (mmscfd), Gas collection status | **Supporting Infrastructure Dataset:** Facility operational metadata for interpreting satellite emission variability. | **B (Supporting)** |
| **Carbon Mapper Airborne Plumes** | `carbonmapper_ch4_plumelist_2020_2021.xls` | 2020–2021 | 2,527 plume rows, 8 columns | Coordinates, Flight candidate IDs, Instantaneous emission rate ($q_{plume}$ in kg/h), Uncertainty | **Supporting Spatial Dataset:** Ultra-high resolution (~5m) airborne validation of satellite plume footprints. | **B (Supporting)** |
| **New York Mobile Monitoring Surveys** | `DATASETS (1).zip -> Data.zip` | 2021–2022 campaigns | >312,000 rows (14 CSVs) | Continuous GPS, Ambient $\text{CH}_4$ (ppm/ppb), 2m air temp, station pressure, sonic wind speed, solar radiation | **Supporting Dispersion Dataset:** High-frequency validation of surface gas plume dispersion under micrometeorological forcing. | **B (Supporting)** |
| **Climate TRACE Waste Geometries** | `waste_geometries.gpkg` | Static GIS layer | 62,662 points globally | Worldwide landfill/dumpsite point coordinates, annual modeled emissions | **Supporting Spatial Mask:** Global GIS boundary layer to filter non-landfill fires from satellite thermal data. | **B (Supporting)** |

---

## 4. Datasets Excluded from the Project

To maintain academic defensibility and eliminate redundancy, the following files are excluded from core processing:
1. **`Landfill_plume_detections (2).csv`:** Exact 100% duplicate of `Landfill_plume_detections.csv`.
2. **`carbonmapper_ch4_plumelist_2020_2021 (1).xls`:** Exact 100% duplicate of `carbonmapper_ch4_plumelist_2020_2021.xls`.
3. **`p3.pdf` & `p4.pdf`:** Duplicate copies of the Nature (2025) and ISPRS (2024) papers already present under full titles.
4. **`Global Identification...Using.pdf` & `Thermal hotspot detection in landfills.pdf`:** Unedited draft duplicates of published articles.
5. **`The toxic impacts of microplastics (MPs)...on marine bivalves.pdf`:** Marine clam toxicology study accidentally included in the folder; completely unrelated to landfills.

---

## 5. Feasibility of Fire-Risk Prediction: What Can Be Honestly Claimed

The literature and dataset audit established a vital scientific distinction that must be communicated clearly to your supervisor:

### What Is Fully Feasible & Defensible:
* **Methane Hotspot Detection:** High-resolution spaceborne and airborne sensors (GHGSat, PRISMA, Carbon Mapper) successfully isolate localized point-source plumes with verified mass emission rates.
* **Surface Thermal Anomaly Mapping:** Satellite thermal infrared sensors (MODIS, Landsat LST) successfully identify persistent high-temperature zones on landfills.
* **Subsurface Elevated-Temperature Fire-Risk Classification:** Using the Bridgeton (`k6`) and Ohio ETLF datasets, we can train robust supervised classifiers (Random Forest, XGBoost, Multilayer Perceptron) to categorize wellhead conditions into Low, Medium, and High fire/reaction risk with expected test accuracy exceeding 90%.

### What Is NOT Feasible with Current Data (And Why):
* **Forward Temporal Fire Prediction (Forecasting Outbreaks Days in Advance):**  
  True predictive forecasting ($P(\text{Fire at } t+\Delta t \mid X_t)$) requires continuous pre-fire sensor time series leading up to confirmed ignition timestamps, alongside confirmed non-fire control periods.  
  Because satellite methane plumes are intermittent clear-sky snapshots (revisited once every few weeks), and MODIS fire points are unlinked to pre-fire sensor streams, claiming a "validated predictive future fire model" is scientifically unsupportable.

### The Defensible Solution:
Frame the project contribution as an **AI-Based Multi-Source Landfill Hazard Screening and Fire-Risk Assessment System**. The system identifies active surface gas/thermal hotspots and computes a transparent, composite risk score based on empirical reaction chemistry and meteorological vulnerability.

---

## 6. Proposed Research Novelty

Based on the 32 reviewed research papers, the project’s novelty rests on three solid pillars:
1. **Bridging the Remote Sensing and In-Situ Monitoring Divide:** Previous papers either examine satellite plumes without ground thermal/chemical data (Nature 2025, ES&T 2025) OR examine in-situ gas well temperatures without spatial remote sensing (Waste Management 2021). Our framework is among the first to synthesize spaceborne plume evidence with validated wellhead smouldering reaction thresholds.
2. **Multi-Source Empirical Consistency Benchmarking:** Comparing satellite-observed point emissions against regulatory models (EPA LMOP / LandGEM) and demonstrating how environmental covariates (temperature, barometric pumping) explain the massive observed discrepancies.
3. **Explainable and Interpretable Risk Prioritization:** Implementing SHAP (SHapley Additive exPlanations) and causal cut-sets from Fuzzy Fault Tree Analysis to make risk scores transparent and actionable for municipal landfill operators.

---

## 7. Immediate Recommended Next Steps

1. **Phase 1 (Week 1–2): Ground-Truth Fire Risk Model Training**  
   Train and benchmark Random Forest, XGBoost, and MLP classifiers on the 18,469 rows of the Bridgeton `k6` dataset. Evaluate multiclass metrics (Precision, Recall, F1-score) for predicting Low, Medium, and High subsurface reaction risk. Validate against the Ohio ETLF dataset.
2. **Phase 2 (Week 3–4): Satellite Hotspot & Thermal Feature Extraction**  
   Implement spatial buffer scripts (1.5 km radius) in Python using `waste_geometries.gpkg` and `landfills_hyper_obs.xlsx` to extract 2025 MODIS thermal anomalies and satellite methane plume densities over target Indian landfills (Ghazipur, Bhalswa, Deonar).
3. **Phase 3 (Week 5–6): Multi-Source Integration & Risk Engine Deployment**  
   Develop the composite hazard-scoring algorithm that fuses satellite plume intensity, surface thermal persistence, and meteorological covariates (ambient temperature and pressure drops) into an interactive visualization dashboard.
