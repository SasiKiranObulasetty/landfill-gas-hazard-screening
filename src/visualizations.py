"""
visualizations.py - Scientific charting and interactive visualization module.
AI-Based Hazardous Gas Hotspot Screening in Solid Waste Landfills.

Provides both interactive Plotly figures for Streamlit UI and publication-quality
Matplotlib figures for report generation.

Strict scientific requirements:
- Gas-composition plots are NEVER represented as geographic landfill maps.
- All axes have units (Vol %, deg F, Ratio, Counts).
- Explains sampling limitations on temporal trends.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Attempt to import plotly; provide graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Consistent research color palette
COLOR_MAP = {
    'Low/Normal': '#2ca02c',  # Green
    'Medium': '#ff7f0e',      # Orange
    'High': '#d62728'         # Red
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'


# ==============================================================================
# 1. GAS CONCENTRATION DISTRIBUTIONS
# ==============================================================================
def create_gas_distributions_figure(df: pd.DataFrame, use_plotly: bool = True):
    """Creates distribution plots for CH4, CO2, and O2."""
    if use_plotly and HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            "Methane (CH4 Vol %)", "Carbon Dioxide (CO2 Vol %)", "Oxygen (O2 Vol %)"
        ))
        
        for cat in ['Low/Normal', 'Medium', 'High']:
            sub = df[df['Risk_Category'] == cat]
            if len(sub) == 0:
                continue
            color = COLOR_MAP.get(cat, '#333')
            
            fig.add_trace(go.Histogram(
                x=sub['CH4_percent'], name=cat, marker_color=color, opacity=0.6,
                legendgroup=cat, showlegend=True
            ), row=1, col=1)
            
            fig.add_trace(go.Histogram(
                x=sub['CO2_percent'], name=cat, marker_color=color, opacity=0.6,
                legendgroup=cat, showlegend=False
            ), row=1, col=2)
            
            fig.add_trace(go.Histogram(
                x=sub['O2_percent'], name=cat, marker_color=color, opacity=0.6,
                legendgroup=cat, showlegend=False
            ), row=1, col=3)

        fig.update_layout(
            barmode='overlay',
            title="Gas Concentration Distributions by Risk Category (N=18,469)",
            height=400,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="CH4 (%)", row=1, col=1)
        fig.update_xaxes(title_text="CO2 (%)", row=1, col=2)
        fig.update_xaxes(title_text="O2 (%)", row=1, col=3)
        fig.update_yaxes(title_text="Observation Count", row=1, col=1)
        return fig
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for cat in ['Low/Normal', 'Medium', 'High']:
            sub = df[df['Risk_Category'] == cat]
            color = COLOR_MAP.get(cat, '#333')
            axes[0].hist(sub['CH4_percent'].dropna(), bins=30, alpha=0.5, color=color, label=cat)
            axes[1].hist(sub['CO2_percent'].dropna(), bins=30, alpha=0.5, color=color, label=cat)
            axes[2].hist(sub['O2_percent'].dropna(), bins=30, alpha=0.5, color=color, label=cat)

        axes[0].set_title("Methane (CH4 Vol %)", fontweight='bold')
        axes[0].set_xlabel("CH4 (Vol %)")
        axes[0].set_ylabel("Frequency Count")
        axes[0].legend()

        axes[1].set_title("Carbon Dioxide (CO2 Vol %)", fontweight='bold')
        axes[1].set_xlabel("CO2 (Vol %)")

        axes[2].set_title("Oxygen (O2 Vol %)", fontweight='bold')
        axes[2].set_xlabel("O2 (Vol %)")

        plt.suptitle("Gas Concentration Distributions Grouped by Target Risk Category", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig


# ==============================================================================
# 2. CH4 vs CO2 PHYSICAL SEPARABILITY SCATTER PLOT
# ==============================================================================
def create_ch4_co2_scatter_figure(df: pd.DataFrame, use_plotly: bool = True):
    """
    Creates the core physical separability chart: CH4 vs CO2.
    Strictly documents that this is chemical coordinate space, not a geographic map.
    """
    if use_plotly and HAS_PLOTLY:
        # Sample down if large for smooth web rendering
        plot_df = df.sample(n=min(len(df), 5000), random_state=42) if len(df) > 5000 else df
        
        fig = px.scatter(
            plot_df,
            x='CH4_percent',
            y='CO2_percent',
            color='Risk_Category',
            color_discrete_map=COLOR_MAP,
            category_orders={'Risk_Category': ['Low/Normal', 'Medium', 'High']},
            hover_data=['Well_Name', 'Date_Sampled', 'Temperature_F', 'O2_percent'],
            opacity=0.6,
            title="Physical Gas Separability in (CH4, CO2) Chemical Space<br><sup>Note: This is a chemical composition diagram, NOT a geographic map of the landfill.</sup>"
        )
        
        # Add high risk bounding zone
        fig.add_shape(
            type="rect",
            x0=0, x1=35, y0=50, y1=75,
            fillcolor="red", opacity=0.1,
            line=dict(color="red", width=1, dash="dash")
        )
        fig.add_annotation(
            x=17.5, y=70,
            text="High Risk Zone (CH4 < 35%, CO2 > 50%)<br>Severe Subsurface Reaction Front",
            showarrow=False,
            font=dict(color="red", size=11)
        )
        
        fig.update_layout(
            xaxis_title="Methane Concentration (CH4 Vol %)",
            yaxis_title="Carbon Dioxide Concentration (CO2 Vol %)",
            template="plotly_white",
            height=550
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for cat in ['Low/Normal', 'Medium', 'High']:
            sub = df[df['Risk_Category'] == cat]
            ax.scatter(sub['CH4_percent'], sub['CO2_percent'], color=COLOR_MAP[cat],
                       label=f"{cat} (N={len(sub):,})", alpha=0.4, edgecolors='none', s=20)

        ax.axvspan(0, 35, color='red', alpha=0.08)
        ax.axhspan(50, 75, color='red', alpha=0.08)
        ax.text(17.5, 71, "High Risk Zone\n(CH4 < 35%, CO2 > 50%)", color='darkred',
                ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title("Physical Gas Separability in (CH4, CO2) Space\n(Chemical Inversion of Subsurface Reaction State)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Methane Concentration (CH4 Vol %)", fontweight='semibold')
        ax.set_ylabel("Carbon Dioxide Concentration (CO2 Vol %)", fontweight='semibold')
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 80)
        ax.legend(frameon=True, facecolor='white')
        plt.tight_layout()
        return fig


# ==============================================================================
# 3. CLASS DISTRIBUTION BAR CHART
# ==============================================================================
def create_class_distribution_figure(df: pd.DataFrame, use_plotly: bool = True):
    """Bar chart showing class counts and proportions."""
    counts = df['Risk_Category'].value_counts().reindex(['Low/Normal', 'Medium', 'High']).fillna(0)
    pcts = (counts / len(df)) * 100

    if use_plotly and HAS_PLOTLY:
        fig = go.Figure(data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                text=[f"{v:,} ({p:.1f}%)" for v, p in zip(counts.values, pcts.values)],
                textposition='auto',
                marker_color=[COLOR_MAP[c] for c in counts.index]
            )
        ])
        fig.update_layout(
            title="Target Risk Category Class Distribution (N=18,469)",
            xaxis_title="Risk Category (Temperature-Derived Heuristic)",
            yaxis_title="Total Observations",
            template="plotly_white",
            height=400
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color=[COLOR_MAP[c] for c in counts.index], width=0.55)
        for b, p in zip(bars, pcts):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 150, f"{int(h):,} ({p:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title("Target Risk Category Distribution", fontweight='bold')
        ax.set_ylabel("Number of Observations")
        ax.set_ylim(0, max(counts.values) * 1.15)
        plt.tight_layout()
        return fig


# ==============================================================================
# 4. TEMPORAL TRENDS
# ==============================================================================
def create_temporal_trends_figure(df: pd.DataFrame, use_plotly: bool = True):
    """
    Monthly time-series of category counts over time.
    Explains sampling limitations in the subtitle/caption.
    """
    df_temp = df.copy()
    df_temp['YearMonth'] = df_temp['Date_Sampled'].dt.to_period('M').dt.to_timestamp()
    monthly = df_temp.groupby(['YearMonth', 'Risk_Category']).size().unstack(fill_value=0)
    for c in ['Low/Normal', 'Medium', 'High']:
        if c not in monthly.columns:
            monthly[c] = 0

    if use_plotly and HAS_PLOTLY:
        fig = go.Figure()
        for cat in ['Low/Normal', 'Medium', 'High']:
            fig.add_trace(go.Scatter(
                x=monthly.index,
                y=monthly[cat],
                name=cat,
                mode='lines+markers',
                line=dict(color=COLOR_MAP[cat], width=2)
            ))
        fig.update_layout(
            title="Monthly Risk Category Observations Over Time (2013 - 2016)<br><sup>Note: Monitoring frequency varied between 7 and 30 days; counts reflect both reaction dynamics and sampling schedules.</sup>",
            xaxis_title="Sampling Month",
            yaxis_title="Monthly Wellhead Measurements",
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for cat in ['Low/Normal', 'Medium', 'High']:
            ax.plot(monthly.index, monthly[cat], label=cat, color=COLOR_MAP[cat], marker='o', markersize=3, linewidth=2)
        ax.set_title("Monthly Risk Category Observations (2013 - 2016)\n(Sampling frequency varied; reflects monitoring campaigns)", fontweight='bold', fontsize=11)
        ax.set_xlabel("Sampling Date", fontweight='semibold')
        ax.set_ylabel("Monthly Observations", fontweight='semibold')
        ax.legend()
        plt.tight_layout()
        return fig


# ==============================================================================
# 5. CONFUSION MATRIX HEATMAP
# ==============================================================================
def create_confusion_matrix_figure(cm: List[List[int]], model_name: str = "Random Forest", use_plotly: bool = True):
    """Generates an annotated 3x3 confusion matrix heatmap."""
    classes = ['Low/Normal', 'Medium', 'High']
    cm_arr = np.array(cm)

    if use_plotly and HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=cm_arr,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=cm_arr,
            texttemplate="%{text:,}",
            textfont={"size": 14}
        ))
        fig.update_layout(
            title=f"{model_name} Confusion Matrix (Chronological Test N=3,611)",
            xaxis_title="Predicted Risk Category",
            yaxis_title="True Risk Category (Temperature-Derived)",
            template="plotly_white",
            height=450,
            width=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cax = ax.matshow(cm_arr, cmap='Blues')
        fig.colorbar(cax)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(classes, fontweight='semibold')
        ax.set_yticklabels(classes, fontweight='semibold')

        for i in range(3):
            for j in range(3):
                val = cm_arr[i, j]
                color = 'white' if val > cm_arr.max() / 2 else 'black'
                ax.text(j, i, f"{val:,}", ha='center', va='center', color=color, fontweight='bold', fontsize=11)

        ax.set_title(f"{model_name} Confusion Matrix", fontweight='bold', pad=15)
        ax.set_xlabel("Predicted Risk Category", fontweight='semibold')
        ax.set_ylabel("True Risk Category", fontweight='semibold')
        plt.tight_layout()
        return fig


# ==============================================================================
# 6. FEATURE IMPORTANCE BAR CHART
# ==============================================================================
def create_feature_importance_figure(feat_dict: Dict[str, float], use_plotly: bool = True):
    """Bar chart of model feature importances."""
    features = list(feat_dict.keys())
    importances = list(feat_dict.values())
    
    # Sort ascending
    idx = np.argsort(importances)
    s_feat = [features[i] for i in idx]
    s_imp = [importances[i] for i in idx]

    if use_plotly and HAS_PLOTLY:
        fig = go.Figure(go.Bar(
            x=s_imp,
            y=s_feat,
            orientation='h',
            marker_color='#1f77b4',
            text=[f"{v:.3f}" for v in s_imp],
            textposition='auto'
        ))
        fig.update_layout(
            title="Random Forest Feature Importance (3-Gas Model)",
            xaxis_title="Relative Gini Importance",
            yaxis_title="Sensor Feature",
            template="plotly_white",
            height=350
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(s_feat, s_imp, color='#1f77b4', height=0.5)
        for i, v in enumerate(s_imp):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=9)
        ax.set_title("Random Forest Feature Importance", fontweight='bold')
        ax.set_xlabel("Relative Importance")
        ax.set_xlim(0, max(s_imp) * 1.25)
        plt.tight_layout()
        return fig


# ==============================================================================
# 7. MODEL COMPARISON CHART
# ==============================================================================
def create_model_comparison_figure(comparison_df: pd.DataFrame, use_plotly: bool = True):
    """Bar chart comparing candidate models across metrics."""
    if use_plotly and HAS_PLOTLY:
        fig = go.Figure()
        models = comparison_df['Model'].unique()
        
        for m in ['Accuracy', 'Macro-F1', 'High-Risk Recall']:
            if m in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=m,
                    x=models,
                    y=comparison_df[m],
                    text=[f"{v:.3f}" for v in comparison_df[m]],
                    textposition='auto'
                ))
                
        fig.update_layout(
            barmode='group',
            title="Model Performance Comparison (Chronological Test Set N=3,611)",
            yaxis_title="Metric Score (0.0 - 1.0)",
            template="plotly_white",
            height=420,
            yaxis=dict(range=[0, 1.1])
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        # fallback
        comparison_df.set_index('Model')[['Accuracy', 'Macro-F1', 'High-Risk Recall']].plot(
            kind='bar', ax=ax, width=0.7, colormap='viridis'
        )
        ax.set_title("Model Comparison on Chronological Holdout", fontweight='bold')
        ax.set_ylabel("Score (0.0 to 1.0)")
        ax.set_ylim(0, 1.15)
        ax.legend(loc='lower right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig


# ==============================================================================
# 8. 7-DAY STATE TRANSITION HEATMAP
# ==============================================================================
def create_transition_matrix_figure(trans_df: pd.DataFrame, use_plotly: bool = True):
    """
    Displays the empirical 7-day risk category transition matrix,
    illustrating why forecasting fails due to quenching interventions.
    """
    if use_plotly and HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=trans_df.values,
            x=trans_df.columns.tolist(),
            y=trans_df.index.tolist(),
            colorscale='YlOrRd',
            text=np.round(trans_df.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 13}
        ))
        fig.update_layout(
            title="Empirical 7-Day Risk State Transitions (%)<br><sup>58.9% of High-Risk wells transition to Low/Normal within 7 days due to human valve adjustments.</sup>",
            xaxis_title="State at Time t + 7 Days",
            yaxis_title="State at Time t",
            template="plotly_white",
            height=420,
            width=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cax = ax.matshow(trans_df.values, cmap='YlOrRd')
        fig.colorbar(cax, label='Transition Rate (%)')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(trans_df.columns.tolist(), fontweight='semibold')
        ax.set_yticklabels(trans_df.index.tolist(), fontweight='semibold')

        for i in range(3):
            for j in range(3):
                val = trans_df.values[i, j]
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color=color, fontweight='bold', fontsize=11)

        ax.set_title("Empirical 7-Day Risk State Transitions (%)\n(Quenching intervention phenomenon)", fontweight='bold', pad=15)
        ax.set_xlabel("State at Time t + 7 Days", fontweight='semibold')
        ax.set_ylabel("State at Time t", fontweight='semibold')
        plt.tight_layout()
        return fig
