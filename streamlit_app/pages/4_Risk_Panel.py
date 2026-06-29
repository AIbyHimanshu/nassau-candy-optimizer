"""
Page 4 — Risk & Impact Panel
Profit sensitivity analysis, high-risk alerts, and KPI tracking.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from optimizer import load_artifacts, generate_recommendations, add_risk_labels
from download_data import PRODUCT_FACTORY_MAP, FACTORIES

st.set_page_config(page_title="Risk & Impact Panel", page_icon="⚠️", layout="wide")
st.title("⚠️ Risk & Impact Panel")
st.markdown(
    "Profit sensitivity, high-risk reassignment warnings, and overall "
    "optimisation KPIs for Nassau Candy's factory network."
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def get_processed():
    return pd.read_csv("data/nassau_processed.csv")

@st.cache_resource
def get_model():
    return load_artifacts()

@st.cache_data
def get_recs(_model, _encoders, _features):
    df = get_processed()
    recs = generate_recommendations(df, _model, _encoders, _features, top_n=30)
    return add_risk_labels(recs)

try:
    df = get_processed()
    model, encoders, feature_names = get_model()
    recs = get_recs(model, encoders, feature_names)
except FileNotFoundError:
    st.error("Pipeline artifacts not found. Run the pipeline first (see Home page).")
    st.stop()

# ── KPI overview ───────────────────────────────────────────────────────────────
avg_lead_time     = df["Lead Time"].mean()
avg_distance      = df["Distance Miles"].mean()
avg_margin        = df["Profit Margin"].mean() * 100 if "Profit Margin" in df.columns else 0
total_orders      = len(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Lead Time (days)",   f"{avg_lead_time:.1f}")
c2.metric("Avg Shipping Distance",  f"{avg_distance:.0f} mi")
c3.metric("Avg Profit Margin",      f"{avg_margin:.1f}%")
c4.metric("Total Orders",           f"{total_orders:,}")

st.markdown("---")

# ── High-risk alerts ───────────────────────────────────────────────────────────
st.subheader("🔴 High-Risk Reassignment Alerts")
if not recs.empty:
    high_risk = recs[recs["Risk Level"] == "High"]
    if high_risk.empty:
        st.success("✅ No high-risk reassignments detected.")
    else:
        for _, row in high_risk.iterrows():
            st.warning(
                f"**{row['Product']}** → reassign to **{row['Recommended Factory']}** "
                f"adds **{row['Distance Delta (miles)']:+.0f} miles** but saves "
                f"**{row['Lead Time Reduction (days)']:.1f} days**. "
                f"Verify profitability before executing."
            )
else:
    st.info("No recommendations generated yet.")

st.markdown("---")

# ── Lead time distribution by factory ─────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Lead Time Distribution by Factory")
    fig_box = px.box(
        df, x="Factory", y="Lead Time",
        color="Factory",
        title="Lead Time Spread per Factory",
        labels={"Lead Time": "Lead Time (days)"},
        points="outliers",
    )
    fig_box.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig_box, use_container_width=True)

with col_r:
    st.subheader("Profit Margin by Division")
    if "Profit Margin" in df.columns:
        margin_df = (
            df.groupby("Division")["Profit Margin"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "Avg Margin", "std": "Std Dev"})
        )
        margin_df["Avg Margin"] *= 100
        margin_df["Std Dev"]   *= 100
        fig_margin = px.bar(
            margin_df, x="Division", y="Avg Margin",
            error_y="Std Dev",
            color="Division",
            title="Avg Profit Margin (%) by Division",
            labels={"Avg Margin": "Avg Profit Margin (%)"},
        )
        fig_margin.update_layout(showlegend=False)
        st.plotly_chart(fig_margin, use_container_width=True)
    else:
        st.info("Profit Margin column not available.")

# ── Profit impact vs lead time reduction bubble chart ─────────────────────────
if not recs.empty:
    st.subheader("Risk–Reward Map")
    fig_bubble = px.scatter(
        recs,
        x="Lead Time Reduction (%)",
        y="Profit Impact Score",
        size=recs["Lead Time Reduction (days)"].clip(lower=0.1),
        color="Risk Level",
        color_discrete_map={"Low": "#1D9E75", "Medium": "#F0B429", "High": "#E85D24"},
        hover_name="Product",
        hover_data=["Current Factory", "Recommended Factory"],
        title="Risk–Reward: Lead Time Reduction vs Profit Impact",
        labels={
            "Lead Time Reduction (%)": "Lead Time Reduction (%)",
            "Profit Impact Score":     "Profit Impact Score (higher = safer)",
        },
        height=420,
    )
    # Quadrant lines
    fig_bubble.add_hline(y=recs["Profit Impact Score"].median(), line_dash="dot",
                          line_color="gray", opacity=0.5)
    fig_bubble.add_vline(x=recs["Lead Time Reduction (%)"].median(), line_dash="dot",
                          line_color="gray", opacity=0.5)
    st.plotly_chart(fig_bubble, use_container_width=True)

# ── Orders & lead time trend over time ────────────────────────────────────────
st.subheader("Lead Time Trend Over Time")
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    monthly = (
        df.groupby(df["Order Date"].dt.to_period("M"))["Lead Time"]
        .mean()
        .reset_index()
    )
    monthly["Order Date"] = monthly["Order Date"].astype(str)
    fig_trend = px.line(
        monthly, x="Order Date", y="Lead Time",
        title="Average Lead Time by Month",
        labels={"Lead Time": "Avg Lead Time (days)", "Order Date": "Month"},
    )
    fig_trend.update_traces(line_color="#378ADD")
    st.plotly_chart(fig_trend, use_container_width=True)

# ── Risk summary table ────────────────────────────────────────────────────────
if not recs.empty:
    st.markdown("---")
    st.subheader("Risk Level Summary")
    risk_summary = (
        recs.groupby("Risk Level")
        .agg(
            Count=("Product", "count"),
            Avg_Reduction=("Lead Time Reduction (days)", "mean"),
            Avg_Profit=("Profit Impact Score", "mean"),
        )
        .round(3)
        .reset_index()
    )
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

# ── Model performance ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Model Performance (Training Run)")
metrics_path = "data/model_metrics.csv"
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    fig_m = px.bar(
        metrics_df.reset_index().rename(columns={"index": "Model"}),
        x="Model", y=["RMSE", "MAE"],
        barmode="group",
        title="RMSE & MAE by Model",
        labels={"value": "Error (days)", "variable": "Metric"},
    )
    st.plotly_chart(fig_m, use_container_width=True)
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("model_metrics.csv not found. Run model_train.py first.")
