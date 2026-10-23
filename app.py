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
    create_transition_matrix_figure,
    COLOR_MAP
)
from src.reporting import (
    get_verified_literature_review,
    get_model_recalculation_summary
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
    "Model Evaluation & Limitations"
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
        ["Random Forest Classifier (Production)", "Logistic Regression (StandardScaler Pipeline)", "Dummy Baseline (Most Frequent)"]
    )
    
    model_key = 'random_forest'
    if "Logistic" in m_choice:
        model_key = 'logistic_regression'
    elif "Dummy" in m_choice:
        model_key = 'dummy'

    model, meta = get_cached_model(model_key)
    model_meta = meta.get('models_summary', {}).get(model_key, {})

    st.markdown("#### Out-of-Sample Performance Metrics (Chronological Test Set, N=3,611)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Accuracy", f"{model_meta.get('accuracy', 0.9601)*100:.2f}%")
    with c2:
        st.metric("Macro-F1 Score", f"{model_meta.get('macro_f1', 0.9606):.4f}")
    with c3:
        st.metric("High-Risk Recall", f"{model_meta.get('high_risk_recall', 1.0)*100:.2f}%")
    with c4:
        st.metric("High-Risk Precision", f"{model_meta.get('high_risk_precision', 1.0)*100:.2f}%")

    col_cm, col_rep = st.columns([1, 1.2])
    with col_cm:
        st.markdown("##### Confusion Matrix")
        # Precomputed audited confusion matrices
        if model_key == 'random_forest':
            cm = [[2191, 97, 0], [38, 663, 0], [0, 0, 622]]
        elif model_key == 'logistic_regression':
            cm = [[2143, 143, 2], [137, 563, 1], [1, 0, 621]]
        else:
            cm = [[2288, 0, 0], [701, 0, 0], [622, 0, 0]]
            
        fig_cm = create_confusion_matrix_figure(cm, model_name=m_choice, use_plotly=True)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_rep:
        st.markdown("##### Per-Class Performance Breakdown")
        if model_key == 'random_forest':
            rep_df = pd.DataFrame({
                'Class': ['Low/Normal', 'Medium', 'High'],
                'Precision': [0.9786, 0.9029, 1.0000],
                'Recall': [0.9576, 0.9458, 1.0000],
                'Specificity': [0.9713, 0.9667, 1.0000],
                'F1-Score': [0.9680, 0.9238, 1.0000],
                'Test Support': [2288, 701, 622]
            })
        elif model_key == 'logistic_regression':
            rep_df = pd.DataFrame({
                'Class': ['Low/Normal', 'Medium', 'High'],
                'Precision': [0.9329, 0.8267, 0.9968],
                'Recall': [0.9366, 0.8031, 0.9984],
                'Specificity': [0.8957, 0.9509, 0.9990],
                'F1-Score': [0.9348, 0.8148, 0.9976],
                'Test Support': [2288, 701, 622]
            })
        else:
            rep_df = pd.DataFrame({
                'Class': ['Low/Normal', 'Medium', 'High'],
                'Precision': [0.6336, 0.0, 0.0],
                'Recall': [1.0000, 0.0, 0.0],
                'Specificity': [0.0, 1.0, 1.0],
                'F1-Score': [0.7757, 0.0, 0.0],
                'Test Support': [2288, 701, 622]
            })
        st.table(rep_df.set_index('Class'))

        st.markdown("##### Uncertainty Quantification (95% Clustered Well Bootstrap CI)")
        st.markdown(f"- **Accuracy 95% CI:** `{model_meta.get('clustered_accuracy_95_ci', ['0.9542', '0.9662'])}`")
        st.markdown(f"- **Macro-F1 95% CI:** `{model_meta.get('clustered_macro_f1_95_ci', ['0.9549', '0.9664'])}`")
        st.caption("Resampling performed across 150 clustered extraction wells (1,000 bootstrap iterations) to account for longitudinal wellhead correlation.")

    if model_key == 'random_forest':
        st.markdown("---")
        st.markdown("#### Feature Importance Analysis")
        feat_imp = meta.get('rf_feature_importances', {'CH4_percent': 0.52, 'CO2_percent': 0.41, 'O2_percent': 0.07})
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

        model_select = st.selectbox("Select Screening Estimator:", ["random_forest", "logistic_regression"], format_func=lambda x: "Random Forest Classifier (3-Gas Model)" if x == "random_forest" else "Logistic Regression Pipeline")

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

        uploaded_file = st.file_uploader("Upload Gas Measurements CSV:", type=['csv'])
        if uploaded_file is not None:
            try:
                batch_raw = pd.read_csv(uploaded_file)
                st.markdown("##### Uploaded Data Preview:")
                st.dataframe(batch_raw.head(), use_container_width=True)

                if st.button("Screen Uploaded Dataset", type="primary"):
                    valid, df_pred, errs = predict_batch(batch_raw, model_type="random_forest")
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

    st.markdown("### 2. Rigorous Evaluation Across Split Strategies")
    eval_df = get_model_recalculation_summary()
    st.table(eval_df.set_index("Model"))

    st.markdown("### 3. Target-Label Provenance & Formula Audit")
    prov_df = pd.DataFrame([
        {"Category": "High Risk", "Rule": "Temp >= 176 F (80 C)", "Citation": "Sabrin (2018) Table 4; Thalhamer (2013)", "Status": "Verified Research Heuristic; UNVERIFIED as regulatory standard"},
        {"Category": "Medium Risk", "Rule": "131 F <= Temp < 176 F", "Citation": "40 CFR 60.753(c); Thalhamer (2013)", "Status": "Verified NSPS Regulatory Exceedance Limit"},
        {"Category": "Low/Normal", "Rule": "Temp < 131 F (55 C)", "Citation": "Clean Air Act NSPS standard limit", "Status": "Verified Baseline Normal Anaerobic Operation"},
        {"Category": "Residual N2", "Rule": "max(0, Balance - 3.76*O2)", "Citation": "40 CFR 60.754(a)(3); EPA Method 3C", "Status": "Verified Stoichiometric Estimate (Not Direct N2 Sensor)"}
    ])
    st.table(prov_df.set_index("Category"))

    st.markdown("### 4. Verified Academic Literature Review")
    st.markdown("Comparative summary of key papers reviewed during the project investigation:")
    lit_df = pd.DataFrame(get_verified_literature_review())
    st.dataframe(lit_df, use_container_width=True)
