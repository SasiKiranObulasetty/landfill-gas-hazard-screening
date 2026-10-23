# Confirmed Minimal Dataset for Final-Year Project

**Project Title:** AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills  
**Directory:** `C:\Users\obula\OneDrive\Desktop\4th_year_project\Confirmed_Minimal_Datasets`  
**Target Objective:** Subsurface Smoldering Combustion and Thermal Runaway Hazard Classification  

---

## 1. Selected Primary Dataset

The primary dataset confirmed for model development is:
- **File:** [`primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx`](file:///c:/Users/obula/OneDrive/Desktop/4th_year_project/Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx)
- **Original Path:** `Datasets/datasets.zip -> datasets/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx`
- **File Size:** 1,049,378 bytes
- **SHA-256 Hash:** `6e20340b94d7106dfcf39258a248abc243253006baaaf4f537e79493f84ef6e7`
- **Records:** Exactly 18,469 rows and 10 columns.
- **Completeness:** 100% verified (0 null values, 0 duplicate rows).

---

## 2. Dataset Schema & Field Descriptions

| Column Name | Data Type | Physical Meaning | Measurement Unit | Verified Range |
| :--- | :--- | :--- | :--- | :--- |
| `Well_Name` | String | Gas extraction well tag | Identifier | 150 unique wells (e.g. `GEW-70`, `GEW-14`) |
| `Date_Sampled` | DateTime | Sampling timestamp | YYYY-MM-DD HH:MM | 2013-06-01 to 2016-10-31 |
| `CH4_percent` | Float | Methane concentration | % by volume | 10.0% to 58.0% |
| `CO2_percent` | Float | Carbon dioxide concentration | % by volume | 20.0% to 65.0% |
| `O2_percent` | Float | Oxygen concentration | % by volume | 0.0% to 15.0% (Air intrusion indicator) |
| `Residual_N2_percent` | Float | Balance / Nitrogen gas | % by volume | 0.5% to 45.0% (Combustion air marker) |
| `Temperature_F` | Float | Wellhead gas temperature | °F | 61.4°F to 215.0°F |
| `Temperature_C` | Float | Wellhead gas temperature | °C | 16.3°C to 101.7°C |
| `Ratio_CH4_CO2` | Float | Biological health ratio | Dimensionless | 0.14 to 1.66 ($<1.0$ indicates thermal pyrolysis) |
| `Risk_Category` | String | **Target Hazard Class** | Categorical | `Low/Normal` (60.4%), `Medium` (22.1%), `High` (17.5%) |

---

## 3. Label Provenance & Target Leakage Prevention

### Label Provenance
`Risk_Category` represents regulatory ground truth established during the Missouri Department of Natural Resources (MDNR) and US EPA Region 7 oversight of the Bridgeton Landfill subsurface smoldering event:
- **`Low/Normal` (11,157 rows):** Normal anaerobic digestion ($T < 131^\circ\text{F}$, $CH_4 > 45\%$, $CH_4/CO_2 \ge 1.0$, $N_2 \le 15\%$).
- **`Medium` (4,074 rows):** Elevated temperature biological transition ($131^\circ\text{F} \le T < 176^\circ\text{F}$, $N_2 \approx 12-22\%$, $CH_4/CO_2 \approx 0.64-1.24$).
- **`High` (3,238 rows):** Active subsurface smoldering combustion reaction ($T \ge 176^\circ\text{F} / 80^\circ\text{C}$, $N_2 > 20\%$, $CH_4/CO_2 < 0.70$).

### Target Leakage Warning & Approved Modelling Strategy
A key finding of the Phase 8 audit is that **`High` risk is assigned deterministically whenever Temperature_F >= 176.0°F**.
- **The Pitfall:** If `Temperature_F` or `Temperature_C` is provided as an input feature to predict contemporary `Risk_Category`, the machine learning model trivially memorizes the temperature threshold ($T \ge 176^\circ\text{F}$).
- **The Approved Scientific Solution:**
  1. **Option A (Gas-Only Inversion Problem):** Predict `Risk_Category` using **ONLY gas composition** ($CH_4$, $CO_2$, $O_2$, Residual $N_2$, $CH_4/CO_2$) without providing temperature, evaluating whether gas chemistry can detect thermal reactions when downhole thermocouples fail.
  2. **Option B (Lead-Time Reaction Forecasting):** Predict whether well $w$ will transition to `Medium` or `High` risk **$k$ days into the future** ($t+7\text{ days}$ or $t+14\text{ days}$) using historical measurements up to time $t$.

---

## 4. Directory Layout

```
Confirmed_Minimal_Datasets/
├── README.md                                         # This documentation
├── Dataset_Confirmation_Report.md                    # In-depth technical confirmation report
├── primary/
│   └── k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx  # Verified primary dataset (18,469 rows)
├── validation/
│   └── README.md                                     # Documentation of within-Bridgeton validation
└── manifests/
    ├── Confirmed_Dataset_Inventory.csv               # SHA-256 and size verification manifest
    └── Excluded_Candidates.md                        # Technical justification for excluded datasets
```

---

## 5. How to Load and Use the Data (Python Snippet)

```python
import pandas as pd

# Load the confirmed primary dataset
df = pd.read_excel("Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx")

# Verify row count and zero nulls
assert len(df) == 18469
assert df.isnull().sum().sum() == 0

# Ensure chronological ordering for leak-free temporal split
df['Date_Sampled'] = pd.to_datetime(df['Date_Sampled'])
df = df.sort_values(by=['Date_Sampled', 'Well_Name']).reset_index(drop=True)

# Define feature sets
gas_features = ['CH4_percent', 'CO2_percent', 'O2_percent', 'Residual_N2_percent', 'Ratio_CH4_CO2']
target = 'Risk_Category'

# Temporal split (June 2013 - June 2015 Train; July 2015 - Feb 2016 Val; March 2016 - Oct 2016 Test)
train_df = df[df['Date_Sampled'] < '2015-07-01']
val_df = df[(df['Date_Sampled'] >= '2015-07-01') & (df['Date_Sampled'] < '2016-03-01')]
test_df = df[df['Date_Sampled'] >= '2016-03-01']

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
```
