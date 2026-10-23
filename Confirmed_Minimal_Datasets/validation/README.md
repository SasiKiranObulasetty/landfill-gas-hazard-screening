# Validation Directory: Scientific Status and Strategy

**Directory:** `Confirmed_Minimal_Datasets/validation/`  
**Status:** **DELIBERATELY EMPTY OF SECONDARY DATASET FILES**  

---

## Why is this folder empty?

In strict compliance with Phase 8 instructions:
> *"Copy a supporting dataset into `validation/` only if its role is scientifically justified. If none qualifies, leave the folder empty and document the reason."*

Both potential supporting candidates (`ETLF Gas SciHub Final.xlsx` and `k5_bubanj_full_field_measurements_60_rows.xlsx`) were thoroughly audited and disqualified:
1. **`ETLF Gas SciHub Final.xlsx`** lacks a unified row-wise schema (partitioned across 15 separate figure sheets) and lacks ground-truth `Risk_Category` target labels.
2. **`k5_bubanj...`** has only 60 total rows, lacks target labels, and contains zero positive reaction cases (100% normal baseline).

Copying either dataset into `validation/` would fabricate an illusion of cross-site validation while violating basic principles of label compatibility.

---

## Verified Validation Strategy for Bridgeton

Instead of flawed cross-site merging, the primary dataset ([k6_bridgeton](file:///c:/Users/obula/OneDrive/Desktop/4th_year_project/Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx)) provides **18,469 fully labelled, multi-year observations across 150 unique wells**, enabling two leak-free validation protocols:

### Protocol 1: Out-of-Time Forward Temporal Split (Recommended)
- **Training Set (Years 1–2):** 2013-06-01 to 2015-06-30 (~10,879 rows).
- **Validation Set (Year 3):** 2015-07-01 to 2016-02-29 (~3,745 rows).
- **Test Set (Out-of-Time):** 2016-03-01 to 2016-10-31 (~3,845 rows).
*Rationale:* Reflects real-world operational deployment where historical data trains an AI to predict future unseen well states.

### Protocol 2: Spatial Well Cluster Holdout (`GroupKFold`)
- Group folds by `Well_Name` (150 wells).
- Ensures that wells in the test fold are completely unseen during training, evaluating the model's ability to generalize to new or unmonitored extraction wells across the landfill body.
