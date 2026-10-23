# Phase Five & Limitations: Audit Boundaries, Data Gaps, and Technical Constraints

## 1. Executive Statement on Audit Integrity

This research audit was conducted under strict **read-only protocols**. No existing files, datasets, folder structures, or metadata were modified, overwritten, transformed, or deleted. All analyses were executed using non-destructive local script execution, Python scientific libraries (Python 3.12 stack with `netCDF4`, `pandas`, `sqlite3`, `xarray`), Microsoft Excel COM automation, and internal binary stream decompilation.

This document formally details all unsupported formats, tool constraints, data gaps, temporal mismatches, and scientific boundaries identified during the investigation.

---

## 2. Technical Inspection Constraints & Unsupported File Behaviors

1. **Headless Microsoft Word COM Hanging on PDF Conversion:**
   * *Observation:* Attempting to open large binary PDF research papers via headless Microsoft Word COM automation (`$word.Documents.Open`) caused background processes to hang indefinitely.
   * *Root Cause:* Microsoft Word triggers an interactive modal dialog ("Word will now convert your PDF to an editable Word document...") when processing complex PDFs, which blocks headless execution.
   * *Resolution:* The hung process (`task-126`) was immediately terminated and orphaned `WINWORD` instances were cleaned up. All 39 PDF research papers were successfully inspected using an in-house Python text stream decompressor (`zlib.decompress` on FlateDecode streams) combined with regex metadata extraction (`/Title`, `/Author`, `/Subject`, and XMP XML metadata). Zero papers were omitted.

2. **Console Character Encoding on Windows (CP-1252 vs UTF-8):**
   * *Observation:* Python scripts printing European author names or scientific subscripts (e.g., Polish characters in `k5.pdf`, subscripts in $\text{CH}_4$) failed with `UnicodeEncodeError: 'charmap' codec can't encode character...`.
   * *Resolution:* All audit scripts and file generators were explicitly configured with `encoding='utf-8'` and UTF-8 Byte Order Mark (`utf-8-sig`) handling for PowerShell JSON outputs.

3. **Archived ZIP Container Inspection:**
   * *Observation:* Research datasets were encapsulated within nested archives (`Datasets/DATASETS (1).zip` containing `Data.zip`, and `655880d7fc235e0aa45f8b8cbbf140f5.zip` containing ERA5 NetCDFs).
   * *Resolution:* Unzipping was performed strictly in-memory using `io.BytesIO` streams or extracted into an isolated scratch cache outside the source repository. Source archives remained completely untouched.

---

## 3. Fundamental Data Gaps in the Available Collection

### Gap A: Absence of Synchronized Ground-Truth Landfill Fire Labels on Satellite Plume Dates
* *Critical Finding:* There is **zero temporal overlap** between the high-resolution satellite methane plume detections (2020–2022) and the nationwide MODIS active fire detections (2025).
* *Impact:* One cannot directly train a supervised model to test whether an observed satellite methane plume on date $T$ erupted into a satellite-detected fire on date $T + \Delta t$ using the current files alone.

### Gap B: Atmospheric Methane Plumes vs Ground-Truth Subsurface Fires
* *Physical Decoupling:* Satellite instruments (GHGSat, PRISMA, EMIT) detect *atmospheric column enhancements* of methane escaping into the air. Subsurface landfill fires, however, occur *inside the anaerobic waste mass* (at depths of 5 to 30 meters).
* *Combustion Chemistry:* As established by Paper `k3` (STOTEN 2021) and Paper `k6` (Waste Management 2021), when an active subsurface fire or smouldering combustion front is initiated, microbial methanogens are killed by high heat (>70–80°C) and methane is actively consumed as fuel. Consequently, **methane concentrations and emissions plummet**, while $\text{CO}_2$, $\text{CO}$ (surging past 1000 ppm), and $\text{H}_2$ spike.
* *Scientific Warning:* A naive machine learning model trained to equate "highest methane plume" with "immediate fire outbreak" will fail physically. High methane indicates active anaerobic biological digestion with poor gas collection; a fire outbreak often presents as an abrupt drop in methane ratio combined with extreme localized thermal radiance.

### Gap C: Coarse Spatial Resolution of MODIS Thermal Anomalies
* *Resolution Limit:* MODIS active fire pixels have a nominal ground footprint of $1\text{ km} \times 1\text{ km}$ at nadir (expanding up to $2\text{ km} \times 4.8\text{ km}$ at scan edges).
* *Consequence:* A single MODIS fire pixel encloses the entire landfill and adjacent industrial estates, highways, or agricultural plots. Without spatial verification, external fires (e.g. crop residue burning adjacent to Ghazipur or Bhalswa) will produce false-positive landfill fire labels.

### Gap D: Clear-Sky Sampling Bias of Satellite Remote Sensing
* *Observation Constraint:* Spaceborne optical and short-wave infrared (SWIR) spectrometers cannot penetrate clouds. During the South Asian summer monsoon (June–September), optical satellite observations over Indian landfills are almost entirely unavailable, precisely when humidity and biological decomposition peak.

---

## 4. Methodological Distinction: What Is Feasible vs What Is Infeasible

To ensure scientific defensibility before your faculty supervisor, the four core project tasks must be formally distinguished:

```
+---------------------------------------------------------------------------------------------------+
| TASK LEVEL                        | FEASIBILITY WITH CURRENT DATA | SCIENTIFIC DEFICIENCY / BLOCKER |
+---------------------------------------------------------------------------------------------------+
| 1. Methane Hotspot Detection      | FULLY FEASIBLE (HIGH CONFIDENCE)| None. Multiple high-res       |
|                                   |                               | datasets available (GHGSat,    |
|                                   |                               | Carbon Mapper, Hyper-Landfill)|
+---------------------------------------------------------------------------------------------------+
| 2. Historical Thermal Anomaly     | FULLY FEASIBLE (HIGH CONFIDENCE)| None. MODIS 2025 India covers  |
|    Screening & Spatial Analysis   |                               | full year; joinable with       |
|                                   |                               | landfill coordinates (Ghazipur)|
+---------------------------------------------------------------------------------------------------+
| 3. Multi-Factor Fire-Risk         | FULLY FEASIBLE (HIGH CONFIDENCE)| None. Supervised ML trained on |
|    Assessment & Classification    |                               | Bridgeton (k6, 18k rows) &     |
|                                   |                               | ETLF (>20k rows) with RSET     |
+---------------------------------------------------------------------------------------------------+
| 4. Future Fire-Risk Prediction    | CURRENTLY INFEASIBLE (LOW)    | Requires synchronized forward  |
|    (Forward Temporal Forecasting) |                               | fire-event timestamps and      |
|                                   |                               | continuous pre-fire sensor feed|
+---------------------------------------------------------------------------------------------------+
```

### Why "Future Fire Prediction" Cannot Be Honestly Claimed:
1. **Lack of Verified Non-Fire vs Pre-Fire Continuous Time-Series:** To train a forward predictive model ($P(\text{Fire at } t + 7\text{ days} \mid X_t)$), one requires continuous sensor streams preceding confirmed fire ignition timestamps across multiple sites, alongside true negative non-fire operating periods.
2. **Data Leakage Risk:** Training a classifier on contemporaneous thermal anomalies (predicting fire from FRP or high brightness temperature) is **historical fire detection / symptom identification**, NOT future prediction. Predicting fire from the heat of the fire itself is circular data leakage.
3. **Severe Class Imbalance:** Verified catastrophic landfill fires are rare discrete events (occurring zero to 3 times per year at a given site), whereas daily observations are overwhelmingly non-fire. Without rigorous event logs, model evaluation metrics (e.g., accuracy) are deceptive.

---

## 5. Minimum Data Requirement to Bridge the Fire Prediction Gap

To upgrade the framework from **Fire-Risk Assessment** to **True Predictive Fire Forecasting**, the following single targeted download is required:

* **Target Dataset:** Historical VIIRS 375m Active Fire Detections (VNP14IMGTDL / VJ114IMGTDL) from NASA FIRMS.
* **Coverage Needed:** Bounding boxes surrounding 5 major Indian landfills (Ghazipur, Bhalswa, Okhla, Deonar, Brahmapuram) covering the years **2021 to 2024**.
* **Why Required:** VIIRS provides $375\text{ m}$ spatial resolution (9x finer than MODIS 1 km) and spans the exact observation years of the satellite methane plume catalogues (2021–2024), enabling exact temporal matching with documented major fire outbreaks (e.g., the March 2022 Ghazipur fire and March 2023 Brahmapuram fire).
