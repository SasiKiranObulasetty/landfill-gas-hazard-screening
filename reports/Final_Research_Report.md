# AI-Based Hazardous Gas Hotspot Screening and Subsurface Thermal State Classification in Solid Waste Landfills

**A Comprehensive Scientific Investigation and Empirical Evaluation**

**Author / Researcher:** Final-Year Engineering Research Project  
**Institution:** Department of Computer Science & Environmental Engineering  
**Academic Year:** 2025–2026  
**Primary Dataset:** Bridgeton Landfill Compliance Monitoring Records (Missouri Department of Natural Resources, 2013–2016)  
**Artifact Repository:** `Confirmed_Minimal_Datasets/`, `experiments/`, `src/`, `models/`, `reports/`

---

## Executive Summary

Subsurface smoldering events (SSEs) in solid waste landfills represent an insidious, environmentally catastrophic geo-environmental hazard characterized by underground exothermic pyrolysis ($176^\circ\text{F} - 240^\circ\text{F}$ / $80^\circ\text{C} - 115^\circ\text{C}$) of buried municipal refuse. Traditional monitoring relies on invasive thermocouple probes that frequently fail under corrosive leachate and thermal stress. This research presents an artificial intelligence screening methodology that inverts subterranean thermal reaction states directly from routine surface-measured landfill gas telemetry—specifically **Methane ($\text{CH}_4$)**, **Carbon Dioxide ($\text{CO}_2$)**, and **Oxygen ($\text{O}_2$)**. 

Through an exhaustive source-traceable scientific audit of 18,469 longitudinal field observations across 150 extraction wells at the Bridgeton Landfill Superfund quarry, we demonstrate that a Random Forest classifier trained strictly on the 3 primary sensor gases achieves **96.01% holdout accuracy (95% Clustered Well CI: 95.42% – 96.62%) and 100.0% recall on severe thermal reactions** ($N=622$ holdout test cases). When evaluated on a strict out-of-sample benchmark of **30 physically unseen extraction wells** over an 8-month future monitoring horizon, the 3-gas model achieves **95.74% accuracy and 100.0% High-Risk recall**. Thermodynamic coordinate analysis confirms that this near-perfect recall is grounded in the irreversible biological cessation of methanogenesis ($\text{CH}_4 < 35\%$) and surge in oxidation products ($\text{CO}_2 > 50\%$) under thermal stress.

Crucially, this report documents a decisive scientific boundary: while contemporaneous classification is strongly supported, **7-day predictive forecasting fails decisively** (35.79% – 45.79% accuracy, consistently underperforming the 43.16% – 52.48% naive persistence baselines). Forensic state-transition analysis reveals that in 58.9% of instances, high-risk well readings reverted to normal within 7 days in the monitoring logs. While active field interventions (vacuum throttling, soil capping) have been hypothesized in literature to explain such rapid state changes, independent operational logs are not present in this compliance dataset to verify this hypothesis. The empirical fact remains that high temporal volatility in routine compliance logs renders static 7-day predictive forecasting unviable. The AI system is therefore implemented as an **in-situ wellhead screening and anomaly localization tool**, not an autonomous fire forecasting system.

---

## 1. Abstract

Subsurface thermal reactions and smoldering combustion within municipal solid waste (MSW) landfills generate hazardous atmospheric emissions, degrade basal containment liners, and accelerate slope instability. Detection has historically required invasive downhole temperature sensors vulnerable to mechanical shear and acidic corrosion. This research designs, validates, and deploys a non-invasive machine learning screening architecture that estimates downhole temperature-derived risk categories directly from three routinely monitored wellhead gases: $\text{CH}_4$, $\text{CO}_2$, and $\text{O}_2$. Utilizing 18,469 verified compliance monitoring records collected under regulatory orders at Bridgeton Landfill, Missouri, we evaluate Dummy, Logistic Regression, and Random Forest estimators across chronological, grouped-well, and spatio-temporal holdout splits. 

Random Forest demonstrates 96.01% out-of-sample accuracy and 100.0% recall for severe reaction states ($T \ge 176^\circ\text{F} / 80^\circ\text{C}$), with clustered well bootstrap 95% confidence intervals demonstrating robust spatial stability ([95.42%, 96.62%]). In contrast, seven-day lead-time forecasting yields only 35.79%–45.79% accuracy, failing to outperform naive persistence due to high empirical state volatility in monitoring records; while operator quenching actions (e.g., vacuum throttling or capping) have been hypothesized in technical literature as drivers of rapid thermal reversals, they remain unverified due to the absence of operational intervention logs in regulatory compliance records. A modular Python and Streamlit web application is delivered for interactive single-wellhead and batch CSV screening, complete with physical input validation, uncertainty quantification, and strict scientific disclaimers.

---

## 2. Introduction and Research Motivation

Municipal solid waste generation globally exceeds 2.1 billion metric tons annually, with over 70% deposited directly into sanitary or semi-engineered landfills. Within anaerobic containment cells, microbial degradation of organic refuse produces landfill gas (LFG), typically composed of 50–55% methane ($\text{CH}_4$), 40–45% carbon dioxide ($\text{CO}_2$), and trace volatile organic compounds (VOCs). 

Under normal operational conditions, this biological digestion operates mesophilically ($95^\circ\text{F} - 105^\circ\text{F}$) or mildly thermophilically ($120^\circ\text{F} - 130^\circ\text{F}$). However, when atmospheric oxygen is inadvertently drawn into deep waste layers—typically caused by excessive vacuum extraction applied by gas recovery blowers—aerobic oxidation is initiated. Because municipal refuse possesses high thermal insulation ($k \approx 0.3 - 0.5\text{ W/m}\cdot\text{K}$) and low hydraulic permeability, heat generated by oxidation cannot readily dissipate. The internal temperature progressively climbs beyond the lethal threshold of methanogens ($>140^\circ\text{F} / 60^\circ\text{C}$), transitioning into self-sustaining, micro-aerophilic smoldering combustion at temperatures exceeding $176^\circ\text{F} - 240^\circ\text{F}$ ($80^\circ\text{C} - 115^\circ\text{C}$).

These subsurface reactions present unique operational and societal crises:
1. **Structural Settlement & Liner Rupture:** High temperatures melt high-density polyethylene (HDPE) geomembrane bottom liners (melting point $\approx 266^\circ\text{F} / 130^\circ\text{C}$), allowing toxic leachate to contaminate regional aquifers.
2. **Hazardous Atmospheric Emissions:** Incomplete anaerobic pyrolysis generates large volumes of toxic carbon monoxide ($\text{CO}$), hydrogen sulfide ($\text{H}_2\text{S}$), sulfur dioxide ($\text{SO}_2$), benzene, and noxious volatile thiols.
3. **Catastrophic Surface Flare-Ups:** Underground smoldering voids can collapse suddenly, creating sinkholes that breach the soil cap, admitting atmospheric air and triggering open surface fires.

Existing monitoring practices rely on routine manual measurements with portable infrared gas analyzers (e.g., Landtec GEM-2000/5000) and downhole thermocouple wires. Thermocouples, however, are prone to rapid drift, wire shearing, and chemical degradation. In contrast, wellhead gas sampling is non-invasive, mandatory under regulatory compliance programs, and performed weekly across every extraction wellhead. 

**Core Research Question:** *Can machine learning models learn a reliable, reproducible chemical mapping from surface-monitored macro-gas concentrations ($\text{CH}_4, \text{CO}_2, \text{O}_2$) to downhole thermal risk categories without suffering from feature leakage or spatial overfitting?*

---

## 3. What is a Solid Waste Landfill?

A modern engineered sanitary landfill is a civil infrastructure system designed to isolate municipal, commercial, and industrial solid waste from the surrounding environment. Unlike uncontrolled open dumpsites, sanitary landfills incorporate multiple engineered barrier layers:
- **Composite Basal Liner:** Typically comprised of a low-permeability compacted clay barrier ($k \le 10^{-7}\text{ cm/s}$) overlain by a 60–80 mil HDPE geomembrane and a non-woven geotextile cushion.
- **Leachate Collection and Removal System (LCRS):** Perforated HDPE pipes bedded within washed gravel layers that continuously drain gravity-fed liquids to sumps for off-site treatment.
- **Active Landfill Gas Extraction Wellfield:** Vertical perforated pipes drilled at 100–200 ft grid spacings, connected via horizontal lateral piping to vacuum blowers and utility flare stacks.
- **Final Capping System:** Multi-layer composite cap consisting of barrier soils, geomembranes, drainage geonets, and topsoil to minimize precipitation infiltration and fugitive emissions.

```
+-------------------------------------------------------------------------+
|                  TYPICAL SANITARY LANDFILL CROSS-SECTION                |
+-------------------------------------------------------------------------+
|                                                                         |
|   Atmospheric Air Intrusion (Vacuum Draw)                              |
|           |                                                             |
|           v                                                             |
|   +-----------------------------------------------------------------+   |
|   | Final Soil & Geomembrane Cap                                    |   |
|   +-----------------------------------------------------------------+   |
|   | Municipal Solid Waste Lift N                                    |   |
|   |  - Normal Digestion Zone: 95-125 deg F (50% CH4, 40% CO2)       |   |
|   |  - Biological Methanogenesis Active                             |   |
|   +-----------------------------------------------------------------+   |
|   | Municipal Solid Waste Lift N-1                                  |   |
|   |  - SMOLDERING HOTSPOT ZONE: >176 deg F (CH4 < 35%, CO2 > 50%)   |   |
|   |  - Exothermic Pyrolysis / Methanogenesis Halted                 |   |
|   +-----------------------------------------------------------------+   |
|   | Washed Gravel Leachate Drainage Layer                           |   |
|   +-----------------------------------------------------------------+   |
|   | HDPE Geomembrane Liner (Vulnerable to Thermal Puncture)        |   |
|   +-----------------------------------------------------------------+   |
|   | Low-Permeability Compacted Clay Sub-base                        |   |
+-------------------------------------------------------------------------+
```

---

## 4. Problems Associated with Landfills and Their Societal Impacts

Landfills represent long-term bio-physicochemical reactors that remain active for 30 to 100 years post-closure. The major environmental hazards include:
1. **Uncontrolled Subsurface Smoldering:** Underground smoldering combustion consumes refuse mass, forming subterranean cavities that trigger sudden sinkholes, compromising gas collection headers and endangering heavy machinery.
2. **Toxic Air Emissions:** Landfill smoldering emits dense aerosols of particulates, volatile organic compounds, and organosulfur odorants that trigger severe respiratory distress, eye irritation, and nausea across adjacent residential neighborhoods.
3. **Groundwater Contamination:** Thermally damaged liners release high concentrations of heavy metals (arsenic, lead), volatile chlorinated solvents (vinyl chloride, trichloroethylene), and per- and polyfluoroalkyl substances (PFAS) into regional aquifers.
4. **Economic and Legal Liabilities:** Remediation of active subsurface landfill reactions costs municipalities and facility operators tens of millions of dollars in cooling barrier walls, inert gas injection, and legal settlements.

---

## 5. Existing Technological Solutions and Their Limitations

Landfill operators currently utilize several detection methodologies, each subject to critical technical limitations:

| Technology / Method | Physical Principle | Key Limitations |
| :--- | :--- | :--- |
| **Downhole Thermocouple Probes** | Direct measurement of Seebeck voltage from dual-metal junctions lowered into well casings. | High failure rate ($>40\%$ over 2 years) due to acidic condensation ($\text{pH} < 4$), mechanical shearing during refuse settlement, and high cost of deep replacement. |
| **Infrared Satellite Thermography (MODIS/Sentinel-2)** | Detection of surface thermal infrared anomalies ($\lambda = 10.5 - 12.5\ \mu\text{m}$). | Sensitive only to surface thermal signatures; deep subterranean smoldering ($50 - 200\text{ ft}$ deep) does not exhibit a surface thermal signature through 5–10 ft of soil cover until catastrophic cap breakthrough occurs. |
| **Drone-Based Aerial Gas Sniffers** | UAV-mounted optical sensors detecting surface methane plumes. | Detects fugitive surface leaks, not downhole wellhead gas composition; highly vulnerable to wind dispersion and weather conditions. |
| **Laboratory Gas Chromatography (EPA Method 3C)** | Off-site gas chromatography measuring permanent gases ($H_2, CO, CH_4, CO_2, N_2, O_2$). | Turnaround time of 1 to 3 weeks; cost prohibitive for weekly compliance monitoring across 150 extraction wells. |
| **Proposed Machine Learning Screening Tool** | Non-invasive statistical pattern recognition on routine wellhead gas measurements ($CH_4, CO_2, O_2$). | Immediate, zero added hardware cost, robust out-of-sample localization; acts as an automated early-warning screening filter. |

---

## 6. Literature Review and Research Gap

An exhaustive review of 27 peer-reviewed papers and technical audit documents was conducted. Below is a structured comparative analysis of the primary relevant literature:

| Ref / Paper | Title & Authors | Methodology | Dataset & Context | Primary Findings | Research Gap & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Paper 1** | *Risk Probability Assessment of Landfill Fire Incidents* | Fuzzy Fault Tree Analysis (FTA) | Operational logs and expert interviews | Maps causal failure chains leading to landfill fires. | Conceptual framework only; no empirical sensor telemetry or ML hotspot detection. |
| **Paper 4** | *Satellite Survey of Methane Emissions from 60 US Landfills* | High-resolution satellite spectrometry | 60 municipal solid waste landfills in North America | Demonstrates massive discrepancies between facility emission inventories and satellite plume observations. | Remote sensing cannot detect subsurface smoldering; no downhole thermal modeling. |
| **Paper 5** | *Monitoring Open Landfill Fires Using Integrated UAV and Sentinel-2* | Random Forest & Gradient Tree Boosting | 2023 Sarimukti landfill fire event (Indonesia) | Achieves ~81% classification accuracy in mapping post-fire burn severity scars. | Evaluates surface fire aftermath rather than subterranean pre-fire reaction screening. |
| **Paper 6** | *Optimizing LandGEM Model Parameters Using Machine Learning* | K-Nearest Neighbors (KNN) inverse modeling | Nationwide US EPA LandGEM database | Optimizes methane generation rate ($k$) and potential ($L_0$). | Macro-scale annual volume estimation; cannot detect localized wellhead thermal anomalies. |
| **Paper 7** | *Innovative Drone-Based Methodology for Quantifying Landfill Methane* | UAV mass-balance flight transects | Aerial campaigns at Italian municipal landfill | Rapid spatial mapping of fugitive surface methane plumes. | Short duration campaigns; no subsurface thermal reaction front screening. |
| **Paper 10** | *Genetic Algorithm–ANN Modeling for VOC Emissions from Landfills* | Hybrid GA-ANN non-linear regression | 9 months of field data (105 sets) from Beijing landfill | Non-linear modeling captures complex meteorological relationships with volatile emissions. | Surface air quality focus; no downhole smoldering state classification. |
| **Paper 25 / k6** | *Subsurface Landfill Fire Prediction Using ML Approaches* (Sabrin, 2018) | SVM, ANN, Decision Trees, Random Forest | Bridgeton Landfill MDNR compliance monitoring records (2013–2016) | Proposes temperature-derived thresholds ($<131^\circ\text{F}, 131-176^\circ\text{F}, \ge 176^\circ\text{F}$); reports $>95\%$ classification accuracy. | Target labels were synthesized from temperature cutoffs, not confirmed fire events; attempted 7-day forecasting failed due to unobserved operator valve adjustments. |

### Research Gap Identified
Existing literature either investigates macro-level methane generation (LandGEM), post-fire satellite burn mapping (Sentinel-2), or claims predictive fire forecasting without validating against operator interventions or investigating the underlying chemical separability. There is a critical research gap in rigorously auditing whether **simple 3-gas surface measurements ($\text{CH}_4, \text{CO}_2, \text{O}_2$) physically separate subsurface reaction states** and defining the exact mathematical limits of predictive time-series forecasting.

---

## 7. Dataset Selection, Provenance and Description

### 7.1 Site Context: Bridgeton Landfill
The Bridgeton Landfill is a deep, inactive limestone quarry landfill located in Bridgeton, Missouri. In 2010, an active subsurface smoldering event was detected in the South Quarry cell, characterized by downhole temperatures exceeding $200^\circ\text{F} - 240^\circ\text{F}$, rapid waste settlement ($>1\text{ ft/week}$), elevated carbon monoxide ($>1,000\text{ ppm}$), and loss of methane production. The site was subjected to intensive regulatory oversight by the Missouri Department of Natural Resources (MDNR), requiring comprehensive weekly monitoring across its 150 extraction wells.

### 7.2 Dataset Integrity & Header Verification
The primary dataset utilized in this project is archived at:
`Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx`

- **Record Count:** Exactly 18,469 validated longitudinal monitoring observations.
- **Wellfield Scale:** 150 unique gas extraction wells (GEW-001 through GEW-150).
- **Date Range:** June 1, 2013 to October 31, 2016 (1,248 days / 3.4 years).
- **Measurement Instrumentation:** Direct field monitoring was conducted using calibrated Landtec GEM-2000 and GEM-5000 infrared landfill gas analyzers equipped with dual-wavelength infrared absorption cells ($\text{CH}_4, \text{CO}_2$) and electrochemical fuel cells ($\text{O}_2$). Temperatures were recorded via downhole thermocouple probes.

---

## 8. Proposed Methodology and System Architecture

The project implements a modular, zero-leakage machine learning screening architecture:

```
+-----------------------------------------------------------------------------------+
|                        SYSTEM ARCHITECTURE & PIPELINE                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Raw Bridgeton Monitoring Logs] (18,469 Rows, 150 Wells)                         |
|         |                                                                         |
|         v                                                                         |
|  [src.data_loader] ===> Fast OpenXML Ingestion & Strict Schema Validation         |
|         |                                                                         |
|         v                                                                         |
|  [src.preprocessing] ===> Strict Feature Isolation (Zero Leakage Protocol)        |
|         |                 - Kept: CH4_percent, CO2_percent, O2_percent            |
|         |                 - Excluded: Temp, N2, Ratios, IDs, Timestamps           |
|         |                                                                         |
|         +-----------------------+-----------------------+                         |
|         |                       |                       |                         |
|         v                       v                       v                         |
|   [Chronological Split]    [GroupKFold Split]    [Spatio-Temporal Split]          |
|   (11,247 Tr / 3,611 Te)   (150 Wells, 5 Folds)  (120 Tr Wells / 30 Unseen Wells) |
|         |                       |                       |                         |
|         +-----------------------+-----------------------+                         |
|                                 |                                                 |
|                                 v                                                 |
|  [src.train_model] ===> Model Training & Pipeline Fitting                         |
|         |               - Dummy Most Frequent Baseline                            |
|         |               - Logistic Regression (StandardScaler Pipeline)           |
|         |               - Random Forest (100 Trees, Depth 10, Balanced)           |
|         |                                                                         |
|         v                                                                         |
|  [src.evaluate] ===> Comprehensive Metric Evaluation                              |
|         |             - Accuracy, Macro-F1, Precision, Recall, Specificity        |
|         |             - 1,000-Iteration Clustered Well-Level Bootstrap CIs        |
|         |                                                                         |
|         v                                                                         |
|  [src.predict] & [app.py] ===> Interactive Web Application & Batch Screening     |
+-----------------------------------------------------------------------------------+
```

---

## 9. Data Preprocessing and Target-Label Construction

### 9.1 Target Label Provenance
The target variable `Risk_Category` classifies wellhead thermal reaction states into three categories:
- **Low/Normal Risk ($T < 131^\circ\text{F} / 55^\circ\text{C}$):** Optimal mesophilic/thermophilic methanogenic baseline per Clean Air Act NSPS operating standards (40 CFR §60.753(c)).
- **Medium Risk ($131^\circ\text{F} \le T < 176^\circ\text{F}$):** Regulatory exceedance indicating thermal stress, transition state, or close proximity to an active reaction front.
- **High Risk ($T \ge 176^\circ\text{F} / 80^\circ\text{C}$):** Severe exothermic subsurface reaction front per Thalhamer (2013) and Sabrin (2018).

### 9.2 Zero-Leakage Feature Extraction
To prevent artificial performance inflation, all temperature variables (`Temperature_F`, `Temperature_C`), engineered domain indices (`Combustion_Air_Index`, `Pyrolysis_Index`), stoichiometric proxies (`Residual_N2_percent`), wellhead IDs, and dates were strictly excluded. Models are trained **strictly on the 3 fundamental sensor gases**:
$$\mathbf{x} = [\text{CH}_4\text{ (Vol \%)},\ \text{CO}_2\text{ (Vol \%)},\ \text{O}_2\text{ (Vol \%)}]^T$$

---

## 10. Experimental Design and Evaluation Metrics

Models are evaluated across three distinct splitting strategies:
1. **Chronological Holdout (Exp A):** Evaluates temporal stability across identical wells over time. Training set: June 1, 2013 – June 30, 2015 ($N=11,247$). Validation purge window: 244 days (July 1, 2015 – February 29, 2016). Test set: March 1, 2016 – October 31, 2016 ($N=3,611$).
2. **Grouped 5-Fold Cross-Validation:** Partitions the 150 extraction wells into 5 independent folds of 30 distinct wells each. Guarantees zero wellhead overlap between training and validation.
3. **Combined Spatio-Temporal Holdout:** The strictest out-of-sample benchmark possible: 120 training wells before July 1, 2015 ($N=8,977$) tested against **30 physically unseen extraction wells** after March 1, 2016 ($N=704$). Zero well overlap AND zero date overlap.

### Evaluation Metrics
- **Accuracy:** Overall proportion of correct classifications.
- **Macro-F1 Score:** Unweighted arithmetic mean of per-class F1-scores, penalizing models that ignore minority hazard classes.
- **Per-Class Precision, Recall, and Specificity:** Critical focus on High-Risk recall ($\text{TP} / [\text{TP} + \text{FN}]$) and specificity ($\text{TN} / [\text{TN} + \text{FP}]$).
- **Clustered Well-Level Bootstrap:** Resampling entire physical extraction wells with replacement ($B=1,000$) to generate true spatial 95% confidence intervals that account for longitudinal clustering.

---

## 11. Results and Comparative Model Analysis

### 11.1 Master Performance Recalculation Table

| Experiment & Configuration | Feature Set | Split Type | Test $N$ (High $N$) | Accuracy | Clustered Acc 95% CI | Macro-F1 | Clustered F1 95% CI | High-Risk Recall | High-Risk Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | Prior Class | Chronological | 3,611 (622) | 0.5912 | [0.5749, 0.6083] | 0.2477 | [0.2434, 0.2521] | 0.0000 | 0.0000 |
| **Logistic Regression Pipeline** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | Chronological | 3,611 (622) | 0.9272 | [0.9179, 0.9349] | 0.9301 | [0.9212, 0.9372] | **0.9984** | 0.9968 |
| **Random Forest (Production)** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | Chronological | 3,611 (622) | **0.9601** | **[0.9542, 0.9662]** | **0.9606** | **[0.9549, 0.9664]** | **1.0000** | **1.0000** |
| **Random Forest (Spatio-Temporal)** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | 30 Unseen Wells + Future | 704 (125) | **0.9574** | **[0.9435, 0.9709]** | **0.9591** | **[0.9464, 0.9719]** | **1.0000** | **1.0000** |
| **Random Forest (GroupKFold)** | $\text{CH}_4, \text{CO}_2, \text{O}_2$ | 5-Fold Grouped Well CV | 18,469 (3,238) | **0.9567** | **[0.9542, 0.9593]** | **0.9551** | **[0.9523, 0.9578]** | **0.9997** | **1.0000** |
| **7-Day Forecasting RF (Verified Test)** | 10 Feats at $t$ | Chronological $t+7\text{d}$ | 1,330 (245) | **0.3579** | [0.3320, 0.3840] | **0.2908** | [0.2680, 0.3140] | **0.1388** | 0.1236 |
| **7-Day Forecasting RF (Phase 10 Variation)** | 7 Feats at $t$ | Chronological $t+7\text{d}$ | 1,391 (296) | **0.4579** | [0.4012, 0.5146] | **0.2908** | [0.2310, 0.3510] | **0.0811** | 0.1983 |

### 11.2 Comprehensive Classical Machine Learning Model Comparison (8 Classifiers)

To rigorously test whether alternative classical learning paradigms outperform or match Random Forest, 8 distinct classifiers were trained and evaluated on the exact same chronological holdout split ($N=3,611$ test observations, sampling date $\ge \text{2016-01-01}$, 8-month purge window). All candidate models were evaluated strictly using the 3 non-invasive gas features ($\text{CH}_4, \text{CO}_2, \text{O}_2$), with all preprocessing transformations (such as feature standard scaling for distance and margin methods) fitted exclusively on training data ($N=11,247$).

| Classifier | Preprocessing Pipeline | Accuracy | Macro-F1 | Weighted-F1 | High-Risk Recall | High-Risk Precision | Train Time (s) | Infer Time (s) | Clustered Acc 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Majority Baseline** | None (Prior distribution) | 0.5912 | 0.2477 | 0.4394 | 0.0000 | 0.0000 | 0.1576 | 0.0026 | [0.5749, 0.6083] |
| **Logistic Regression** | StandardScaler + Balanced | 0.9272 | 0.9301 | 0.9284 | 0.9984 | 0.9968 | 0.5795 | 0.0035 | [0.9179, 0.9349] |
| **Random Forest (Production)** | Max Depth 10 + Balanced | **0.9601** | **0.9606** | **0.9601** | **1.0000** | **1.0000** | 1.2966 | 0.1655 | [0.9542, 0.9662] |
| **Decision Tree** | Max Depth 10 + Balanced | 0.9554 | 0.9560 | 0.9554 | **1.0000** | 0.9968 | **0.0771** | **0.0025** | [0.9482, 0.9618] |
| **Extra Trees** | 100 Trees + Balanced | 0.9416 | 0.9435 | 0.9422 | 0.9984 | 0.9968 | 0.6910 | 0.1114 | [0.9338, 0.9488] |
| **HistGradientBoosting** | Iterations 100 + Balanced | 0.9521 | 0.9527 | 0.9521 | **1.0000** | **1.0000** | 4.7580 | 0.0355 | [0.9452, 0.9585] |
| **Support Vector Machine (SVM)** | StandardScaler + RBF Kernel | 0.9358 | 0.9379 | 0.9366 | 0.9984 | 0.9952 | 11.6784 | 1.3609 | [0.9263, 0.9443] |
| **K-Nearest Neighbors (KNN)** | StandardScaler + $k=5$ Distance | 0.9529 | 0.9515 | 0.9524 | 0.9920 | 0.9952 | **0.0503** | 0.0304 | [0.9454, 0.9598] |

#### Key Analytical Insights
1. **Ensemble & Tree Dominance:** Random Forest achieved the highest overall Accuracy (96.01%) and Macro-F1 (0.9606), closely followed by single Decision Tree (95.54%), KNN (95.29%), and HistGradientBoosting (95.21%).
2. **High-Risk Class Sensitivity:** Random Forest, Decision Tree, and HistGradientBoosting all achieved **100.0% recall** on the High-Risk hazard class ($622/622$), validating the strong non-linear boundary formed by methanogenesis shutdown ($\text{CH}_4 < 35\%$, $\text{CO}_2 > 50\%$).
3. **Computational Latency Comparison:** Decision Tree exhibited the lowest inference latency (**2.5 ms** for 3,611 samples; $\sim 0.7\,\mu\text{s}$ per query), making it exceptionally attractive for low-power edge microcontrollers. Conversely, SVM required the longest training time (11.68 s) and highest inference latency (1.36 s) due to quadratic kernel distance computations across large candidate sets.
4. **Baseline Comparison:** All 7 trained classifiers significantly exceeded the zero-information majority baseline (59.12% Accuracy, 0.2477 Macro-F1), establishing that surface gas telemetry contains unambiguous biochemical signatures of subsurface thermal stress.

### 11.3 The Physical Basis of High-Risk Classification
The finding that Random Forest achieves **100.0% recall on High-Risk cases** ($622/622$ in chronological test; $125/125$ on unseen wells) was subjected to intense scrutiny. Coordinate geometry analysis revealed that this performance is driven by near-complete separability in $(\text{CH}_4, \text{CO}_2)$ space:
- **Normal Wells:** Typically $\text{CH}_4 \in [45\%, 58\%]$, $\text{CO}_2 \in [35\%, 45\%]$.
- **High Risk Wells ($T \ge 176^\circ\text{F}$):** In the primary verified dataset, high-risk gas concentrations span $\text{CH}_4 \in [10.10\%, 34.90\%]$ and $\text{CO}_2 \in [50.20\%, 74.70\%]$.
- **Post-Hoc Empirical Boundary:** The rectangular cutoff boundary ($\text{CH}_4 \le 35.0\%$ and $\text{CO}_2 \ge 50.0\%$) was identified through post-hoc exploratory data analysis rather than as an a priori prespecified regulatory standard or universal physical threshold. Across all 18,469 observations, all 3,238 High-Risk records (100.0%) satisfy this boundary, while 15,230 out of 15,231 Non-High-Risk records (99.993%) remain outside. Exactly one non-high-risk observation satisfied this rule: borehole `GEW-45` measured on 2014-08-27, with downhole temperature $134.0^\circ\text{F}$ (Medium risk), $\text{CH}_4 = 34.8\%$, $\text{CO}_2 = 50.1\%$, and $\text{O}_2 = 1.2\%$.
- **Thermodynamic Mechanism:** At temperatures above $140^\circ\text{F}$ ($60^\circ\text{C}$), methanogenic archaea die rapidly, halting methane generation. Concurrently, smoldering oxidation of organic carbon consumes residual oxygen and releases large volumes of carbon dioxide ($C + O_2 \to CO_2$). Consequently, elevated temperatures physically cannot coexist with normal methane levels. Even a linear hyper-plane (Logistic Regression) separates 99.84% of High-Risk instances.

---

## 12. Classification versus Forecasting Findings

A primary scientific contribution of this audit is debunking claims of 7-day lead-time risk forecasting:

### Like-for-Like Recomputed Benchmarks (Matched 7-Day Test Split, N = 1,330)

| Benchmark / Model | Test $N$ | Accuracy | Macro-F1 | High-Risk Recall | Scientific Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 1,330 | **58.42%** | **0.2458** | 0.00% | Trivial baseline; predicts dominant Low/Normal state. |
| **Naive Persistence Baseline ($y_{t+7} = y_t$)** | 1,330 | **43.16%** | **0.3377** | 20.00% | Predicts current well category at time $t$ for $t+7\text{d}$. |
| **Random Forest Forecasting Model (Exp B)** | 1,330 | **35.79%** | **0.2908** | 13.88% | **FAILS. Strictly underperforms naive persistence.** |
| **Phase 10 Audited Variation** | 1,391 | **45.79%** | **0.2908** | 8.11% | Broader boundary matching; Macro-F1 remains collapsed at 0.29. |

### State Transitions and Unverified Operational Hypotheses
State-transition matrix analysis revealed that in **58.9% of observations where a wellhead was High Risk ($T \ge 176^\circ\text{F}$), the state transitioned to Low/Normal within 7 days in the monitoring logs**.

Technical landfill literature suggests that field technicians routinely execute operational interventions (e.g., throttling vacuum valves, injecting inert gas, or sealing geomembrane covers) when temperatures exceed regulatory limits. However, because this compliance monitoring dataset contains **zero records of valve adjustments, vacuum pressures, or maintenance actions**, attributing state changes definitively to operator intervention remains an unverified hypothesis. Empirically, the high state volatility in routine compliance logs demonstrates that static 7-day predictive forecasting is scientifically unviable without synchronized operational telemetry.

---

## 13. Application Implementation

To make the research findings usable by civil and environmental engineers, a production-grade, modular web application was developed in Python using Streamlit:
- **`app.py`:** Core user interface implementing 7 dedicated pages:
  - *Page A — Project Overview:* Problem statement, facility context, scope, and disclaimers.
  - *Page B — Dataset Explorer:* Paginated inspection, schema definitions, statistical summaries, interactive filters, and CSV export.
  - *Page C — Risk-Category Classification:* Model performance metrics, confusion matrices, classification reports, and feature importance.
  - *Page D — Interactive Gas Screening:* Single-wellhead manual input and batch CSV upload with strict range validation and formatted prediction output.
  - *Page E — Research Visualizations:* Interactive Plotly charts for gas distributions, CH4 vs CO2 scatter, temporal trends, and model comparisons.
  - *Page F — Seven-Day Forecasting Research:* Transparent presentation of forecasting failures and state-transition quenching matrices.
  - *Page G — Model Evaluation & Limitations:* Source provenance, leakage audits, and split integrity summaries.
- **Modular Backend (`src/`):** Dedicated modules for data ingestion (`data_loader.py`), zero-leakage preprocessing (`preprocessing.py`), model training and serialization (`train_model.py`), safe inference and validation (`predict.py`), metric calculation (`evaluate.py`), scientific charting (`visualizations.py`), and reporting (`reporting.py`).

---

## 14. Limitations and Threats to Validity

1. **Target Construct Validity:** `Risk_Category` is a reconstructed academic research label based on temperature cutoffs ($<131^\circ\text{F}$, $131-176^\circ\text{F}$, $\ge 176^\circ\text{F}$), not an official regulatory ground truth or confirmed open flame.
2. **Missing Operational Telemetry:** The absence of wellhead valve settings, vacuum pressures, and maintenance intervention logs precludes predictive state-trajectory forecasting.
3. **Lack of 3D Volumetric Coordinates:** Wellhead measurements are taken at surface collars. The model cannot identify the vertical depth ($Z$) of the smoldering reaction front along the perforated casing without downhole temperature logs.
4. **External Landfill Generalizability:** The models were trained exclusively on data from the Bridgeton Landfill quarry. While the underlying biochemical principles of methanogenesis inhibition are universal, quantitative decision boundaries require site-specific calibration before deployment at standard shallow landfills.

---

## 15. Conclusion and Future Work

### 15.1 Concluding Summary
This research project successfully designed, audited, and implemented an AI-based hazardous gas hotspot screening methodology for municipal solid waste landfills. By restricting model inputs strictly to three routinely measured surface gases ($\text{CH}_4, \text{CO}_2, \text{O}_2$), the Random Forest classifier achieves **96.01% out-of-sample accuracy and 100.0% High-Risk recall**, grounded in the irreversible thermodynamic cessation of methanogenesis under thermal stress. The study definitively established that the dataset supports **contemporaneous thermal state classification and wellhead anomaly localization, but does NOT support 7-day predictive forecasting or open surface fire prediction**.

### 15.2 Future Work
1. **Integration of Operational SCADA Telemetry:** Future research should ingest real-time valve position logs, header vacuum pressures, and flare flow rates into recurrent neural architectures (LSTM/GRU) to model operator feedback loops.
2. **Multi-Facility Transfer Learning:** Calibrate and validate the screening model across diverse municipal landfill geometries (e.g., canyon fills, area mounds) to establish universal transfer functions.
3. **Fiber-Optic Distributed Temperature Sensing (DTS):** Correlate surface gas composition with high-resolution vertical temperature profiles to resolve the 3D depth of subterranean smoldering fronts.

---

## 16. References

1. **Sabrin, S. (2018).** *Subsurface Landfill Fire Prediction Using Machine Learning Approaches*. Master's Thesis, Department of Civil and Environmental Engineering, Rowan University. Glassboro, NJ. (`Papers/k6.pdf`).
2. **Thalhamer, T. (2013).** *Subsurface Landfill Fires: Causes, Detection, and Mitigation*. Technical Guidance Report, California Department of Resources Recycling and Recovery (CalRecycle).
3. **Martin, J. W., Stark, T. D., Thalhamer, T., & Giraud, G. (2013).** Detection and monitoring of subsurface smoldering events in municipal solid waste landfills. *Geotechnical and Geological Engineering*, 31(4), 1145–1162.
4. **US Environmental Protection Agency (EPA). (2016).** Standards of Performance for Municipal Solid Waste Landfills. *Code of Federal Regulations*, 40 CFR Part 60, Subpart WWW / Subpart XXX (§60.753(c), §60.754(a)(3)).
5. **Estabrooks, R. (2013).** *Subsurface Reaction Assessment and Environmental Impact at Bridgeton Landfill*. Missouri Department of Natural Resources (MDNR) Technical Report.
6. **Barlaz, M. A., Borden, R. C., & Raga, R. (2016).** Controls on subsurface temperatures and gas composition in landfills experiencing elevated temperatures. *Waste Management*, 53, 110–121.
7. **Jafari, N. H., Stark, T. D., & Merry, S. (2017).** The role of elevated temperatures in landfill slope stability and geomembrane integrity. *Journal of Geotechnical and Geoenvironmental Engineering*, 143(6), 04017015.
8. **Powell, J. T., Townsend, T. G., & Zimmerman, J. B. (2016).** Estimates of solid waste disposal rates and greenhouse gas emissions from US landfills. *Nature Climate Change*, 6(2), 162–165.
