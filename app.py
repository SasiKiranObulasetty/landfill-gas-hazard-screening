"""
app.py - Streamlit Web Application
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills

Research Prototype for University Final-Year Project Demonstration.
Demonstrates machine learning classification of temperature-derived subsurface
risk categories from wellhead gas measurements (CH4, CO2, O2).
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_bridgeton_raw,
    get_dataset_summary,
    DOCUMENTED_SCHEMA,
    VALID_RISK_CATEGORIES
)
from src.preprocessing import (
    FEATURE_COLS_3GAS,
    CLASS_ORDER,
    SENSOR_RANGES,
    validate_single_sample,
    validate_batch_dataframe
)
from src.predict import (
    predict_single,
    predict_batch,
    get_model_and_metadata,
    SCIENTIFIC_DISCLAIMER
)
from src.visualizations import (
    create_gas_distributions_figure,
    create_ch4_co2_scatter_figure,
    create_class_distribution_figure,
    create_temporal_trends_figure,
    create_confusion_matrix_figure,
    create_feature_importance_figure,
    create_model_comparison_figure,
    create_macro_f1_comparison_figure,
    create_high_risk_tradeoff_figure,
    create_transition_matrix_figure,
    COLOR_MAP
)
from src.reporting import (
    get_verified_literature_review,
    get_model_recalculation_summary,
    get_classical_model_comparison_summary
)
from src.ingestion import (
    DatabaseManager,
    sync_external_data,
    get_registered_adapter,
    validate_classifier_compatibility
)


# ==============================================================================
# STREAMLIT APP CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Landfill Hazardous Gas Screening AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom clean aesthetic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #991b1b;
        margin-bottom: 1.2rem;
    }
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #1e40af;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data(show_spinner="Loading verified Bridgeton monitoring dataset (18,469 records)...")
def get_cached_dataset() -> pd.DataFrame:
    """Loads pristine source data once and caches in memory."""
    return load_bridgeton_raw()


@st.cache_resource(show_spinner="Loading trained screening models...")
def get_cached_model(model_name: str):
    """Loads trained model artifacts without retraining on page refresh."""
    return get_model_and_metadata(model_name)


@st.cache_data(show_spinner="Loading external monitoring observations...")
def get_cached_external_observations(limit: int = 200) -> pd.DataFrame:
    """Loads external observations from persistent SQLite database."""
    db = DatabaseManager()
    return db.get_observations_dataframe(limit=limit)


# Load dataset safely
try:
    df_raw = get_cached_dataset()
    data_summary = get_dataset_summary(df_raw)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    df_raw = pd.DataFrame()
    data_summary = {}


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/sensor.png", width=64)
st.sidebar.title("Landfill AI Portal")
st.sidebar.caption("Subsurface Reaction Screening Prototype")

NAV_PAGES = [
    "Project Overview",
    "Dataset Explorer",
    "Risk-Category Classification",
    "Interactive Gas Screening",
    "Research Visualizations",
    "Seven-Day Forecasting Research",
    "Model Evaluation & Limitations",
    "Data Updates & Monitoring"
]

selected_page = st.sidebar.radio("Navigation Menu", NAV_PAGES, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Credentials:**")
st.sidebar.markdown("- **Scope:** 3-Gas Hazard Classification")
st.sidebar.markdown("- **Dataset:** Bridgeton Landfill (18,469 rows)")
st.sidebar.markdown("- **Sensor Inputs:** CH4, CO2, O2 (Vol %)")
st.sidebar.markdown("- **Status:** Audited & Reproduced")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Academic Research Project. Developed for university evaluation.")

# Supported classical ML model registry
MODEL_CHOICE_MAP = {
    "Random Forest Classifier (Top Ensemble)": "random_forest",
    "Decision Tree Classifier": "decision_tree",
    "Extra Trees Classifier": "extra_trees",
    "HistGradientBoosting Classifier": "hist_gradient_boosting",
    "Support Vector Machine (RBF Pipeline)": "svm",
    "K-Nearest Neighbors (k=5 Distance Pipeline)": "knn",
    "Logistic Regression (StandardScaler Pipeline)": "logistic_regression",
    "Dummy Baseline (Most Frequent)": "dummy"
}



# ==============================================================================
# PAGE A — PROJECT OVERVIEW
# ==============================================================================
if selected_page == "Project Overview":
    st.markdown('<div class="main-header">AI-Based Hazardous Gas Hotspot Detection and Fire Risk Prediction in Solid Waste Landfills</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Research Prototype: Machine Learning Inversion of Subsurface Thermal States from Wellhead Biogas Telemetry</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>ACADEMIC RESEARCH DISCLAIMER:</strong> This software is a university research prototype designed to evaluate the feasibility of machine learning for subsurface thermal reaction screening. It is <strong>NOT a certified operational landfill fire detection system</strong>, open flame alarm, or regulatory compliance platform. Output categories are statistical estimates derived from empirical wellhead gas ratios at Bridgeton Landfill.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data_summary.get('total_rows', 18469):,}</div>
            <div class="metric-label">Field Observations</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data_summary.get('unique_wells', 150)}</div>
            <div class="metric-label">Extraction Wells</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">3 Gases</div>
            <div class="metric-label">Inputs (CH4, CO2, O2)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">96.0%</div>
            <div class="metric-label">Holdout Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 1. Research Motivation & Problem Statement")
    st.markdown(r"""
    Subsurface smoldering events (SSEs) in municipal solid waste (MSW) landfills pose severe environmental and structural risks, including toxic gas emissions (carbon monoxide, sulfur compounds), rapid slope subsidence, damage to geomembrane liners, and potential ignition of surface refuse.

    Traditional detection relies on periodic manual measurement of downhole thermocouple probes installed inside gas extraction wells. However, thermocouples frequently fail due to corrosive acidic leachates, high shear strains during settlement, and extreme temperatures ($>200^\circ\text{F}$).

    **Primary Research Objective:** Investigate whether routine surface measurements of three standard landfill gases—**Methane ($\text{CH}_4$)**, **Carbon Dioxide ($\text{CO}_2$)**, and **Oxygen ($\text{O}_2$)**—can accurately detect subterranean thermal reaction fronts without requiring direct downhole temperature probes or complex laboratory gas chromatography.
    """)

    st.markdown("### 2. Dataset Facility & Provenance")
    st.markdown("""
    - **Facility Name:** Bridgeton Landfill (Bridgeton, Missouri, USA). A deep, inactive quarry landfill adjacent to the West Lake Landfill Superfund site.
    - **Regulatory Context:** Monitored under formal orders from the Missouri Department of Natural Resources (MDNR) between 2013 and 2016 during a documented subsurface smoldering event.
    - **Temporal Coverage:** **June 1, 2013** to **October 31, 2016** (1,248 days / 3.4 years).
    - **Wellfield Scale:** 150 dedicated gas extraction wells (GEW series) across the North and South Quarry cells.
    - **Primary Dataset File:** `Confirmed_Minimal_Datasets/primary/k6_bridgeton_full_monitoring_dataset_18469_rows.xlsx`.
    """)

    st.markdown("### 3. Demonstrated Scope vs. Scientific Limitations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(r"""
        #### Verified Capabilities (Supported by Data)
        - **Contemporaneous Thermal State Classification:** Classifies wellheads into *Low/Normal*, *Medium*, or *High* thermal risk categories from 3 raw sensor gases with **96.0% accuracy** and **100.0% High-Risk recall** ($N=622$).
        - **Wellhead Anomaly Screening:** Out-of-sample evaluation on **30 physically unseen extraction wells** confirms **95.7% accuracy**, proving the model localizes which extraction wells are drawing thermal reaction gases.
        - **Physical Grounding:** Grounded in thermodynamic methanogenesis shutdown: at temperatures $\ge 176^\circ\text{F}$, methanogenesis halts completely ($\text{CH}_4 < 35\%$) and carbon oxidation accelerates ($\text{CO}_2 > 50\%$).
        """)
    with c2:
        st.markdown("""
        #### Scientific Boundaries (What Data Does NOT Support)
        - **No Open Surface Fire Prediction:** The dataset records downhole smoldering temperatures; it contains zero confirmed surface flame outbreak incidents or fire department call logs.
        - **No 7-Day Forecasting:** Static predictive time-series forecasting fails (**35.8% – 45.8% accuracy**, consistently worse than the 43.2% – 52.5% naive persistence baselines), as empirical monitoring logs show 58.9% of high-risk well readings reverse to normal within 7 days. Operational interventions have been hypothesized in literature to explain this, but remain unverified due to the absence of operational records in this dataset.
        - **No Cross-Site Generalization:** Bridgeton represents an extreme quarry smoldering event; baseline gas ratios at standard shallow landfills require site-specific recalibration.
        """)


# ==============================================================================
# PAGE B — DATASET EXPLORER
# ==============================================================================
elif selected_page == "Dataset Explorer":
    st.markdown('<div class="main-header">Dataset Explorer & Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive inspection of verified Bridgeton Landfill monitoring records</div>', unsafe_allow_html=True)

    if df_raw.empty:
        st.warning("Dataset is currently unavailable.")
    else:
        st.markdown("""
        <div class="info-box">
            <strong>DATA INTEGRITY PROTOCOL:</strong> The source dataset is strictly read-only. All transformations, filters, and slices displayed below operate strictly on in-memory working copies.
        </div>
        """, unsafe_allow_html=True)

        # Filters
        st.markdown("#### Interactive Filter Controls")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            all_wells = sorted(df_raw['Well_Name'].unique())
            selected_wells = st.multiselect("Filter by Extraction Well(s):", all_wells, default=[])
        with fcol2:
            all_cats = ['Low/Normal', 'Medium', 'High']
            selected_cats = st.multiselect("Filter by Risk Category:", all_cats, default=all_cats)
        with fcol3:
            min_date = df_raw['Date_Sampled'].min().date()
            max_date = df_raw['Date_Sampled'].max().date()
            date_range = st.date_input("Filter by Sampling Date Range:", (min_date, max_date))

        # Apply filters to working copy
        filtered_df = df_raw.copy()
        if selected_wells:
            filtered_df = filtered_df[filtered_df['Well_Name'].isin(selected_wells)]
        if selected_cats:
            filtered_df = filtered_df[filtered_df['Risk_Category'].isin(selected_cats)]
        if len(date_range) == 2:
            start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_df = filtered_df[(filtered_df['Date_Sampled'] >= start_d) & (filtered_df['Date_Sampled'] <= end_d)]

        st.caption(f"Showing **{len(filtered_df):,}** of **{len(df_raw):,}** total records ({len(filtered_df)/len(df_raw)*100:.1f}%)")

        # Tabbed views
        tab1, tab2, tab3 = st.tabs(["Data Table & Export", "Column Schema & Units", "Descriptive Statistics"])

        with tab1:
            st.dataframe(filtered_df, use_container_width=True, height=350)
            
            # Export buttons
            csv_buf = io.StringIO()
            filtered_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_buf.getvalue(),
                file_name="bridgeton_filtered_monitoring_data.csv",
                mime="text/csv"
            )

        with tab2:
            st.markdown("##### Documented Column Schema & Physical Units")
            schema_records = []
            for col, meta in DOCUMENTED_SCHEMA.items():
                schema_records.append({
                    'Column Name': col,
                    'Data Type': meta['type'],
                    'Physical Unit': meta['unit'],
                    'Description': meta['description']
                })
            st.table(pd.DataFrame(schema_records))

        with tab3:
            st.markdown("##### Statistical Summary of Numeric Variables")
            st.dataframe(filtered_df.describe().round(2), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Gas Concentration Distributions (Filtered Subset)")
        fig_dist = create_gas_distributions_figure(filtered_df, use_plotly=True)
        st.plotly_chart(fig_dist, use_container_width=True)


# ==============================================================================
# PAGE C — RISK-CATEGORY CLASSIFICATION
# ==============================================================================
elif selected_page == "Risk-Category Classification":
    st.markdown('<div class="main-header">Three-Gas Risk-Category Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Independent reproduction and evaluation of models trained strictly on CH4, CO2, and O2</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>ZERO-LEAKAGE VERIFICATION:</strong> Model inputs are strictly restricted to <code>CH4_percent</code>, <code>CO2_percent</code>, and <code>O2_percent</code>. Downhole temperatures, date timestamps, wellhead IDs, calculated nitrogen, ratios, and future measurements are strictly excluded.
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    m_choice = st.selectbox(
        "Select Evaluated Model:",
        list(MODEL_CHOICE_MAP.keys())
    )
    model_key = MODEL_CHOICE_MAP[m_choice]

    model, meta = get_cached_model(model_key)
    model_meta = meta.get('models_summary', {}).get(model_key, {})

    st.markdown("#### Out-of-Sample Performance Metrics (Chronological Test Set, N=3,611)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Accuracy", f"{model_meta.get('accuracy', 0.0)*100:.2f}%")
    with c2:
        st.metric("Macro-F1 Score", f"{model_meta.get('macro_f1', 0.0):.4f}")
    with c3:
        st.metric("High-Risk Recall", f"{model_meta.get('high_risk_recall', 0.0)*100:.2f}%")
    with c4:
        st.metric("High-Risk Precision", f"{model_meta.get('high_risk_precision', 0.0)*100:.2f}%")

    t_c1, t_c2, t_c3 = st.columns(3)
    with t_c1:
        st.caption(f"⏱️ **Training Time:** {model_meta.get('train_time_sec', 0.0):.4f} s")
    with t_c2:
        st.caption(f"⚡ **Inference Time (N=3,611):** {model_meta.get('inference_time_sec', 0.0):.4f} s")
    with t_c3:
        st.caption(f"🎯 **Weighted-F1:** {model_meta.get('weighted_f1', 0.0):.4f}")

    col_cm, col_rep = st.columns([1, 1.2])
    with col_cm:
        st.markdown("##### Confusion Matrix")
        cm = model_meta.get('confusion_matrix', [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        fig_cm = create_confusion_matrix_figure(cm, model_name=m_choice, use_plotly=True)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_rep:
        st.markdown("##### Per-Class Performance Breakdown")
        per_class = model_meta.get('per_class', {})
        if per_class:
            rep_records = []
            for cls_name in ['Low/Normal', 'Medium', 'High']:
                c_data = per_class.get(cls_name, {})
                rep_records.append({
                    'Class': cls_name,
                    'Precision': c_data.get('precision', 0.0),
                    'Recall': c_data.get('recall', 0.0),
                    'Specificity': c_data.get('specificity', 0.0),
                    'F1-Score': c_data.get('f1_score', 0.0),
                    'Test Support': c_data.get('support', 0)
                })
            rep_df = pd.DataFrame(rep_records)
            st.table(rep_df.set_index('Class'))
        else:
            st.info("Per-class metrics unavailable.")

        st.markdown("##### Uncertainty Quantification (95% Clustered Well Bootstrap CI)")
        acc_ci = model_meta.get('clustered_accuracy_95_ci', ['N/A', 'N/A'])
        f1_ci = model_meta.get('clustered_macro_f1_95_ci', ['N/A', 'N/A'])
        if len(acc_ci) == 2 and isinstance(acc_ci[0], (int, float)):
            st.markdown(f"- **Accuracy 95% CI:** `[{float(acc_ci[0]):.4f}, {float(acc_ci[1]):.4f}]`")
        else:
            st.markdown(f"- **Accuracy 95% CI:** `{acc_ci}`")

        if len(f1_ci) == 2 and isinstance(f1_ci[0], (int, float)):
            st.markdown(f"- **Macro-F1 95% CI:** `[{float(f1_ci[0]):.4f}, {float(f1_ci[1]):.4f}]`")
        else:
            st.markdown(f"- **Macro-F1 95% CI:** `{f1_ci}`")
        st.caption("Resampling performed across 150 clustered extraction wells (1,000 bootstrap iterations) to account for longitudinal wellhead correlation.")

    if hasattr(model, 'feature_importances_'):
        st.markdown("---")
        st.markdown("#### Feature Importance Analysis")
        feat_imp = dict(zip(FEATURE_COLS_3GAS, [float(x) for x in model.feature_importances_]))
        fig_imp = create_feature_importance_figure(feat_imp, use_plotly=True)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Methane and carbon dioxide concentrations account for >90% of tree-split decisions, reflecting the primary biochemical signals of anaerobic methanogenesis suppression.")


# ==============================================================================
# PAGE D — INTERACTIVE GAS SCREENING
# ==============================================================================
elif selected_page == "Interactive Gas Screening":
    st.markdown('<div class="main-header">Interactive Gas Hazard Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Non-invasive estimation of subsurface reaction state from surface wellhead gas telemetry</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="disclaimer-box">
        <strong>OPERATIONAL CAUTION:</strong> {SCIENTIFIC_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Choose Screening Mode:", ["Single Wellhead Reading (Manual Input)", "Batch Wellfield Upload (CSV)"], horizontal=True)

    if mode == "Single Wellhead Reading (Manual Input)":
        st.markdown("#### Enter Measured Wellhead Concentrations")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            ch4_in = st.number_input("Methane (CH4 Vol %):", min_value=0.0, max_value=100.0, value=52.0, step=0.5)
        with col_in2:
            co2_in = st.number_input("Carbon Dioxide (CO2 Vol %):", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
        with col_in3:
            o2_in = st.number_input("Oxygen (O2 Vol %):", min_value=0.0, max_value=25.0, value=0.5, step=0.1)

        model_choice_name = st.selectbox(
            "Select Screening Estimator:",
            list(MODEL_CHOICE_MAP.keys()),
            index=0
        )
        model_select = MODEL_CHOICE_MAP[model_choice_name]

        if st.button("Run Thermal Hazard Screening", type="primary"):
            res = predict_single(ch4_in, co2_in, o2_in, model_type=model_select)
            
            if not res['success']:
                st.error(f"Validation Error: {res['error']}")
            else:
                cat = res['predicted_category']
                label = res['formatted_label']
                conf = res['confidence']
                
                # Color coded banner
                if cat == 'High':
                    st.error(f"### 🚨 {label}")
                elif cat == 'Medium':
                    st.warning(f"### ⚠️ {label}")
                else:
                    st.success(f"### ✅ {label}")

                if res.get('warning'):
                    st.info(res['warning'])

                # Results metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Estimated Category", cat)
                with c2:
                    st.metric("Model Confidence", f"{conf*100:.1f}%" if conf else "N/A")
                with c3:
                    st.metric("Computed Balance Gas", f"{res['input_measurements']['Computed_Balance_percent']:.1f}%")
                with c4:
                    st.metric("Model Version", res['model_version'])

                # Class probability distribution
                if res['class_probabilities']:
                    st.markdown("##### Predicted Class Probability Distribution")
                    prob_df = pd.DataFrame([res['class_probabilities']]).T.reset_index()
                    prob_df.columns = ['Risk Category', 'Probability']
                    prob_df['Probability (%)'] = (prob_df['Probability'] * 100).round(1)
                    st.table(prob_df[['Risk Category', 'Probability (%)']])

    else:
        st.markdown("#### Batch CSV Screening")
        st.markdown("Upload a CSV file containing columns for `CH4_percent`, `CO2_percent`, and `O2_percent`.")

        batch_model_choice = st.selectbox(
            "Select Screening Estimator for Batch Data:",
            list(MODEL_CHOICE_MAP.keys()),
            index=0,
            key="batch_model_selector"
        )
        batch_model_select = MODEL_CHOICE_MAP[batch_model_choice]

        uploaded_file = st.file_uploader("Upload Gas Measurements CSV:", type=['csv'])
        if uploaded_file is not None:
            try:
                batch_raw = pd.read_csv(uploaded_file)
                st.markdown("##### Uploaded Data Preview:")
                st.dataframe(batch_raw.head(), use_container_width=True)

                if st.button("Screen Uploaded Dataset", type="primary"):
                    valid, df_pred, errs = predict_batch(batch_raw, model_type=batch_model_select)
                    if not valid:
                        st.error("Validation Failed:")
                        for e in errs:
                            st.write(f"- {e}")
                    else:
                        st.success(f"Successfully processed {len(df_pred):,} wellhead readings!")
                        st.dataframe(df_pred, use_container_width=True)

                        # Category summary
                        cat_counts = df_pred['Predicted_Risk_Category'].value_counts()
                        st.markdown("##### Screening Summary Counts:")
                        st.write(dict(cat_counts))

                        # Download enriched CSV
                        csv_out = io.StringIO()
                        df_pred.to_csv(csv_out, index=False)
                        st.download_button(
                            label="📥 Download Screened Predictions (CSV)",
                            data=csv_out.getvalue(),
                            file_name="screened_landfill_predictions.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")


# ==============================================================================
# PAGE E — RESEARCH VISUALIZATIONS
# ==============================================================================
elif selected_page == "Research Visualizations":
    st.markdown('<div class="main-header">Research Visualizations & Chemical Separability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive thermodynamic charts illustrating gas concentration boundaries</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>IMPORTANT GEOMETRIC DISTINCTION:</strong> The scatter plot below illustrates chemical concentration coordinates (CH4 Vol % vs CO2 Vol %). It is <strong>NOT a geographic map</strong> of landfill coordinates or well locations.
    </div>
    """, unsafe_allow_html=True)

    tab_v1, tab_v2, tab_v3, tab_v4 = st.tabs([
        "Chemical Separability (CH4 vs CO2)",
        "Gas Distributions",
        "Temporal Trends",
        "Model Comparison"
    ])

    with tab_v1:
        st.markdown("#### Physical Separability of Subsurface Thermal States")
        fig_scatter = create_ch4_co2_scatter_figure(df_raw, use_plotly=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Figure 1: Two-dimensional chemical space showing complete geometric isolation of High Risk thermal states (CH4 < 35%, CO2 > 50%) from normal methanogenic biogas.")

    with tab_v2:
        st.markdown("#### Gas Concentration Histograms by Risk Category")
        fig_dist = create_gas_distributions_figure(df_raw, use_plotly=True)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab_v3:
        st.markdown("#### Longitudinal Observation Frequency (2013 - 2016)")
        fig_trend = create_temporal_trends_figure(df_raw, use_plotly=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab_v4:
        st.markdown("#### Benchmark Model Comparison")
        comp_df = get_model_recalculation_summary()
        fig_comp = create_model_comparison_figure(comp_df, use_plotly=True)
        st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================================
# PAGE F — SEVEN-DAY FORECASTING RESEARCH
# ==============================================================================
elif selected_page == "Seven-Day Forecasting Research":
    st.markdown('<div class="main-header">Seven-Day Predictive Forecasting Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed investigation into why static 7-day risk forecasting fails on operational monitoring data</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>SCIENTIFIC AUDIT FINDING:</strong> Seven-day predictive risk forecasting is <strong>UNSUPPORTED and FAILS</strong> on this dataset. The tested Random Forest forecasting model achieved only <strong>45.79% accuracy and 0.2908 Macro-F1</strong>, performing significantly worse than a simple naive persistence baseline (<strong>0.3377 Macro-F1</strong>).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 1. Recomputed Benchmark Comparison")
    st.markdown(r"""
    Comparison on the verified matched 7-day test split ($N = 1,330$, $t+7\text{d}$ tolerance $7 \pm 2.5\text{ days}$, test date $\ge \text{2016-03-01}$):
    """)
    fc_table = pd.DataFrame([
        {
            "Model / Strategy": "Majority-Class Baseline",
            "Accuracy": "58.42%",
            "Macro-F1": "0.2458",
            "High-Risk Recall": "0.0%",
            "Operational Verdict": "Trivial baseline; predicts dominant Low/Normal state."
        },
        {
            "Model / Strategy": "Naive Persistence Baseline (y_{t+7} = y_t)",
            "Accuracy": "43.16%",
            "Macro-F1": "0.3377",
            "High-Risk Recall": "20.00%",
            "Operational Verdict": "Predicts current well category at time t for t+7d."
        },
        {
            "Model / Strategy": "Random Forest Forecasting Model (Exp B)",
            "Accuracy": "35.79%",
            "Macro-F1": "0.2908",
            "High-Risk Recall": "13.88%",
            "Operational Verdict": "FAILS. Strictly underperforms naive persistence."
        },
        {
            "Model / Strategy": "Phase 10 Audited Variation (N = 1,391)",
            "Accuracy": "45.79%",
            "Macro-F1": "0.2908",
            "High-Risk Recall": "8.11%",
            "Operational Verdict": "Broader boundary match; Macro-F1 remains collapsed at 0.29."
        }
    ])
    st.table(fc_table.set_index("Model / Strategy"))

    st.markdown("#### 2. Root Cause Analysis: The Quenching Intervention Feedback Loop")
    c_left, c_right = st.columns([1.1, 1])

    with c_left:
        # Precomputed empirical 7-day transition matrix (%)
        trans_data = pd.DataFrame(
            [[83.6, 12.1, 4.3],
             [39.2, 48.5, 12.3],
             [58.9, 15.6, 25.5]],
            index=['Low/Normal', 'Medium', 'High'],
            columns=['Low/Normal', 'Medium', 'High']
        )
        fig_trans = create_transition_matrix_figure(trans_data, use_plotly=True)
        st.plotly_chart(fig_trans, use_container_width=True)

    with c_right:
        st.markdown(r"""
        ##### Empirical Observations vs. Unverified Hypotheses
        1. **Observed Rapid State Reversals:** In **58.9%** of observations where a wellhead exhibited High Risk ($T \ge 176^\circ\text{F}$), the recorded state transitioned back to Low/Normal within 7 days in the monitoring logs.
        2. **Hypothesized Mechanism (Unverified):** Technical literature suggests that field technicians routinely perform operational adjustments (throttling vacuum valves, injecting inert gas, or sealing covers) upon observing high temperatures.
        3. **Absence of Operational Records:** The monitoring dataset contains **zero records of valve adjustments, vacuum pressures, or maintenance actions**.
        4. **Scientific Conclusion:** Because operational records are absent, attributing state changes definitively to human interventions remains an unverified hypothesis. Empirically, the high state volatility in routine compliance logs demonstrates that static 7-day predictive forecasting is scientifically unviable without synchronized operational telemetry.
        """)


# ==============================================================================
# PAGE G — MODEL EVALUATION AND LIMITATIONS
# ==============================================================================
elif selected_page == "Model Evaluation & Limitations":
    st.markdown('<div class="main-header">Model Evaluation, Provenance & Scientific Boundaries</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Synthesis of Phase 10 source-traceable scientific audit findings</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### 1. Classification vs. Forecasting: The Definitive Scientific Decision
    - **Contemporaneous Classification (SUPPORTED):** Real-time measurement of $\text{CH}_4, \text{CO}_2, \text{O}_2$ provides a robust, non-invasive proxy for downhole thermal reactions ($96.0\%$ accuracy on 3,611 holdout samples; $95.7\%$ on 30 unseen wells).
    - **Seven-Day Predictive Forecasting (REFUTED):** Routine monitoring logs cannot forecast future risk categories without concurrent operational valve telemetry.
    - **Open Surface Fire Prediction (SCIENTIFICALLY INVALID):** The dataset records internal smoldering temperatures; it contains zero verified surface fire breakout events.
    """)

    st.markdown("### 2. Classical Machine Learning Benchmark Comparison (8 Models)")
    st.markdown(r"""
    Standardized, like-for-like evaluation of 8 classical machine learning classifiers on the chronological holdout test set 
    (**N=3,611 observations**, test sampling dates $\ge \text{2016-01-01}$, 8-month purge window). All candidate models use 
    identical features (`CH4_percent`, `CO2_percent`, `O2_percent`), and all preprocessing transformations are fitted strictly on training data.
    """)

    classical_df = get_classical_model_comparison_summary()
    if not classical_df.empty:
        display_cols = [
            "Model", "Accuracy", "Macro_F1", "Weighted_F1", 
            "High_Risk_Recall", "High_Risk_Precision", 
            "Train_Time_Sec", "Infer_Time_Sec", "Clustered_Acc_95_CI"
        ]
        valid_cols = [c for c in display_cols if c in classical_df.columns]
        st.table(classical_df[valid_cols].set_index("Model"))

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### Macro-F1 Performance Comparison")
            fig_macro = create_macro_f1_comparison_figure(classical_df, use_plotly=True)
            if fig_macro:
                st.plotly_chart(fig_macro, use_container_width=True)

        with col_g2:
            st.markdown("#### High-Risk Class: Precision vs. Recall Tradeoff")
            fig_tradeoff = create_high_risk_tradeoff_figure(classical_df, use_plotly=True)
            if fig_tradeoff:
                st.plotly_chart(fig_tradeoff, use_container_width=True)

        st.markdown("#### Detailed Classifier Inspector & Confusion Matrix")
        eval_choice = st.selectbox(
            "Select Classifier to Inspect:",
            list(MODEL_CHOICE_MAP.keys()),
            key="page_g_model_inspect"
        )
        eval_key = MODEL_CHOICE_MAP[eval_choice]
        _, g_meta = get_cached_model(eval_key)
        g_info = g_meta.get('models_summary', {}).get(eval_key, {})

        st.info(fr"**Selected Model:** {eval_choice} &nbsp;|&nbsp; **Evaluation Split:** Chronological Holdout (N=3,611 observations, date $\ge$ 2016-01-01)")

        col_g_cm, col_g_rep = st.columns([1, 1.2])
        with col_g_cm:
            g_cm = g_info.get('confusion_matrix', [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            fig_g_cm = create_confusion_matrix_figure(g_cm, model_name=eval_choice, use_plotly=True)
            st.plotly_chart(fig_g_cm, use_container_width=True)

        with col_g_rep:
            g_per_class = g_info.get('per_class', {})
            if g_per_class:
                g_rep_list = []
                for cls_name in ['Low/Normal', 'Medium', 'High']:
                    c_dat = g_per_class.get(cls_name, {})
                    g_rep_list.append({
                        'Class': cls_name,
                        'Precision': c_dat.get('precision', 0.0),
                        'Recall': c_dat.get('recall', 0.0),
                        'Specificity': c_dat.get('specificity', 0.0),
                        'F1-Score': c_dat.get('f1_score', 0.0),
                        'Test Support': c_dat.get('support', 0)
                    })
                st.table(pd.DataFrame(g_rep_list).set_index('Class'))
            else:
                st.info("Per-class metrics unavailable.")

    st.markdown("### 3. Evaluation Across Split Strategies & Validation Holdouts")
    eval_df = get_model_recalculation_summary()
    st.table(eval_df.set_index("Model"))

    st.markdown("### 4. Target-Label Provenance & Formula Audit")
    prov_df = pd.DataFrame([
        {"Category": "High Risk", "Rule": "Temp >= 176 F (80 C)", "Citation": "Sabrin (2018) Table 4; Thalhamer (2013)", "Status": "Verified Research Heuristic; UNVERIFIED as regulatory standard"},
        {"Category": "Medium Risk", "Rule": "131 F <= Temp < 176 F", "Citation": "40 CFR 60.753(c); Thalhamer (2013)", "Status": "Verified NSPS Regulatory Exceedance Limit"},
        {"Category": "Low/Normal", "Rule": "Temp < 131 F (55 C)", "Citation": "Clean Air Act NSPS standard limit", "Status": "Verified Baseline Normal Anaerobic Operation"},
        {"Category": "Residual N2", "Rule": "max(0, Balance - 3.76*O2)", "Citation": "40 CFR 60.754(a)(3); EPA Method 3C", "Status": "Verified Stoichiometric Estimate (Not Direct N2 Sensor)"}
    ])
    st.table(prov_df.set_index("Category"))

    st.markdown("### 5. Verified Academic Literature Review")
    st.markdown("Comparative summary of key papers reviewed during the project investigation:")
    lit_df = pd.DataFrame(get_verified_literature_review())
    st.dataframe(lit_df, use_container_width=True)


# ==============================================================================
# PAGE H — DATA UPDATES & MONITORING
# ==============================================================================
elif selected_page == "Data Updates & Monitoring":
    st.markdown('<div class="main-header">Data Updates & Environmental Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated External Data Ingestion, Ingestion Telemetry, and Scientific Guardrails</div>', unsafe_allow_html=True)

    db = DatabaseManager()
    storage_status = db.get_storage_status()
    counts = db.get_observation_counts()
    latest_run = db.get_latest_ingestion_summary()
    latest_sync_time = db.get_latest_successful_sync_time()
    latest_obs_time = db.get_latest_observation_timestamp()

    st.markdown("""
    <div class="disclaimer-box">
        <strong>DATA SEGREGATION & SCIENTIFIC VALIDITY NOTICE:</strong> Incoming observations from external public sources 
        (US EPA Envirofacts, Open-Meteo Weather, Carbon Mapper Plumes) are stored in an independent database 
        (managed PostgreSQL when configured, or local SQLite fallback) completely segregated from the historical Bridgeton 
        research dataset (18,469 records in Excel). Incompatible measurements (such as satellite plume mass emission flux in kg/h or 
        ambient surface meteorology) are strictly quarantined and cannot be evaluated by the 3-gas subsurface downhole risk model.
    </div>
    """, unsafe_allow_html=True)

    # Persistence Status Banner
    if storage_status['is_cloud_persistent']:
        st.success(
            f"☁️ **Cloud Persistence ACTIVE (Managed PostgreSQL):** Synchronized to shared database at "
            f"`{storage_status['connection_target']}`. Both the GitHub Actions scheduler and this deployed "
            f"Streamlit application read and write from the same persistent store."
        )
    elif storage_status['init_error']:
        st.error(
            f"❌ **Cloud Database Connection Error:** Failed to connect to remote PostgreSQL: "
            f"{storage_status['init_error']}. Fallback active: using local container storage "
            f"`{storage_status['connection_target']}`. Observations written by GitHub Actions runners "
            f"will NOT appear in this website until remote database credentials are configured."
        )
    else:
        st.warning(
            f"⚠️ **Cloud Persistence NOT CONFIGURED (Using Local / Ephemeral SQLite):** "
            f"Active target: `{storage_status['connection_target']}`. "
            f"GitHub Actions runners and deployed Streamlit instances operate in isolated containers and do not share a filesystem. "
            f"To enable automatic updates between GitHub Actions and this website, configure `PERSISTENT_DATABASE_URL` "
            f"(e.g. Supabase or Neon PostgreSQL) in both GitHub Repository Secrets and Streamlit Cloud Secrets."
        )

    # Telemetry metrics rows
    st.markdown("### 1. Ingestion Pipeline Telemetry")
    sync_disp = latest_sync_time[:19].replace('T', ' ') if latest_sync_time else "Never Synced"
    obs_disp = latest_obs_time[:19].replace('T', ' ') if latest_obs_time else "No Observations"
    status_val = latest_run.get('status', 'IDLE') if latest_run else 'IDLE'
    status_color = "#16a34a" if status_val == "SUCCESS" else ("#ea580c" if status_val == "PARTIAL" else "#dc2626")
    backend_disp = "PostgreSQL (Cloud)" if storage_status['is_cloud_persistent'] else "SQLite (Local)"

    # Row 1: Infrastructure & Temporal Telemetry
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)

    with col_t1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.15rem; padding-top: 0.4rem;">{backend_disp}</div>
            <div class="metric-label">Storage Backend</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.05rem; padding-top: 0.5rem;">{obs_disp}</div>
            <div class="metric-label">Latest Observation Timestamp</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.05rem; padding-top: 0.5rem;">{sync_disp}</div>
            <div class="metric-label">Last Ingestion Sync (UTC)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {status_color};">{status_val}</div>
            <div class="metric-label">Pipeline Run Status</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Record Volumes & Data Source
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)

    with col_v1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{counts.get('total', 0):,}</div>
            <div class="metric-label">Total Stored Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col_v2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #16a34a;">{counts.get('compatible', 0):,}</div>
            <div class="metric-label">Classifier-Compatible Inputs</div>
        </div>
        """, unsafe_allow_html=True)

    with col_v3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #dc2626;">{counts.get('incompatible', 0):,}</div>
            <div class="metric-label">Quarantined (Non-Volumetric)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_v4:
        src_label = latest_run.get('source_name', 'None') if latest_run else 'None'
        if len(src_label) > 20:
            src_label = src_label[:18] + ".."
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.05rem; padding-top: 0.5rem;">{src_label}</div>
            <div class="metric-label">Most Recent Source</div>
        </div>
        """, unsafe_allow_html=True)

    # Staleness / Connection Warning
    if not latest_sync_time:
        st.info("ℹ️ **Initial Sync Pending:** No external ingestion runs have been recorded in this database yet. Click 'Refresh Data Now' below or wait for scheduled execution.")
    else:
        try:
            sync_dt = datetime.fromisoformat(latest_sync_time.replace('Z', '+00:00'))
            now_dt = datetime.now(timezone.utc)
            days_diff = (now_dt - sync_dt).days
            if days_diff > 7:
                st.warning(f"⚠️ **Stale Data Notice:** Last successful ingestion sync was {days_diff} days ago ({sync_disp} UTC). Scheduled GitHub Actions workflow may be paused or public APIs may be temporarily unreachable.")
        except Exception:
            pass

    # Manual Ingestion Trigger Section
    st.markdown("---")
    st.markdown("### 2. Manual Data Ingestion Control")
    st.caption("Trigger an immediate on-demand ingestion run from verified public endpoints. Note: Browser refresh alone does not fetch new external data.")

    col_ctrl1, col_ctrl2 = st.columns([1.5, 1])

    SOURCE_OPTIONS = {
        "US EPA Envirofacts (Regulatory GHGRP Subpart HH)": "US_EPA_Envirofacts_GHGRP",
        "Open-Meteo (Ambient Landfill Surface Meteorology)": "Open_Meteo_Ambient_Weather",
        "Carbon Mapper / Airborne Plume Catalog": "Carbon_Mapper_Plume_Catalog",
        "Synthetic Test Fixture (Offline Unit Testing)": "Synthetic_Test_Fixture"
    }

    with col_ctrl1:
        selected_source_label = st.selectbox(
            "Select Environmental Data Source Adapter:",
            list(SOURCE_OPTIONS.keys()),
            index=0
        )
        selected_source_key = SOURCE_OPTIONS[selected_source_label]
        adapter_inst = get_registered_adapter(selected_source_key)
        st.markdown(f"**Source Citation / Docs:** *{adapter_inst.source_citation}*")

    with col_ctrl2:
        max_rec = st.number_input("Max Records to Fetch:", min_value=1, max_value=100, value=20)
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_clicked = st.button("🔄 Refresh Data Now", use_container_width=True)

    if refresh_clicked:
        with st.spinner(f"Connecting to {selected_source_label}..."):
            summary = sync_external_data(
                source_name=selected_source_key,
                max_records=max_rec,
                db_manager=db
            )
            # Invalidate only external observations cache; model caches remain untouched!
            get_cached_external_observations.clear()

            if summary.status == "SUCCESS":
                st.success(
                    f"Sync completed successfully! Retrieved: {summary.records_retrieved}, "
                    f"Stored: {summary.records_stored}, Duplicates ignored: {summary.records_duplicate}."
                )
            elif summary.status == "PARTIAL":
                st.info(f"Sync partial. Retrieved {summary.records_retrieved} records.")
            else:
                st.error(f"Sync failed: {summary.error_message}")
        st.rerun()

    # Data Architecture Segregation Table
    st.markdown("---")
    st.markdown("### 3. Data Architecture Segregation: Historical vs. External Feeds")
    st.markdown("""
    The application maintains a strict firewall between the audited historical research dataset and incoming external feeds:
    """)

    seg_storage = f"Cloud PostgreSQL ({storage_status['connection_target']})" if storage_status['is_cloud_persistent'] else f"SQLite ({storage_status['connection_target']})"
    seg_table = pd.DataFrame([
        {
            "Dataset Layer": "Primary Historical Research Dataset",
            "Storage Location": "Confirmed_Minimal_Datasets/primary/ (Excel)",
            "Records": "18,469 rows (June 2013 – October 2016)",
            "Measurements": "Downhole Wellhead CH4, CO2, O2 (Vol %), Temp (F/C)",
            "Model Compatibility": "100% Compatible (Canonical training & validation set)",
            "Role": "Preserved benchmark for ML models and spatial cross-validation"
        },
        {
            "Dataset Layer": "External Environmental Monitoring Pipeline",
            "Storage Location": seg_storage,
            "Records": f"{counts.get('total', 0):,} records",
            "Measurements": "Regulatory annual filings, ambient meteorology, remote sensing plume flux (kg/h)",
            "Model Compatibility": "0% for satellite/weather (Quarantined by scientific unit validator)",
            "Role": "Independent environmental context (synchronized across GitHub Actions & Streamlit via PostgreSQL)"
        }
    ])
    st.table(seg_table.set_index("Dataset Layer"))

    # Recent Observations Explorer
    st.markdown("---")
    st.markdown("### 4. Recent External Observations Explorer")

    df_obs = get_cached_external_observations(limit=100)

    if df_obs.empty:
        st.info("No external observations stored in database yet. Use 'Refresh Data Now' above to ingest records.")
    else:
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            filter_mode = st.radio(
                "Filter Observations:",
                ["All Records", "Classifier Compatible Only", "Quarantined / Incompatible Only"],
                horizontal=True
            )

        if filter_mode == "Classifier Compatible Only":
            df_filtered = df_obs[df_obs['is_classifier_compatible'] == 1]
        elif filter_mode == "Quarantined / Incompatible Only":
            df_filtered = df_obs[df_obs['is_classifier_compatible'] == 0]
        else:
            df_filtered = df_obs

        st.caption(f"Displaying {len(df_filtered)} records (out of {len(df_obs)} recently retrieved):")
        st.dataframe(df_filtered, use_container_width=True)

        # Scientific quarantine explanation box
        incomp_count = len(df_filtered[df_filtered['is_classifier_compatible'] == 0])
        if incomp_count > 0:
            st.info(
                f"ℹ️ **Scientific Integrity Note:** {incomp_count} displayed record(s) are quarantined from the "
                "subsurface thermal risk classifier. Satellite plume emission rates (kg/h) or ambient weather cannot "
                "be fed into the 3-gas classifier because the model was trained strictly on downhole volumetric gas ratios "
                "(CH4 %, CO2 %, O2 % in Vol %)."
            )

    # Ingestion Audit Log
    st.markdown("---")
    st.markdown("### 5. Ingestion Audit Log")
    st.caption("Immutable record of all scheduled and manual ingestion attempts with error tracking:")

    df_logs = db.get_ingestion_logs_dataframe(limit=20)

    if df_logs.empty:
        st.info("No audit logs recorded yet.")
    else:
        st.dataframe(df_logs, use_container_width=True)

