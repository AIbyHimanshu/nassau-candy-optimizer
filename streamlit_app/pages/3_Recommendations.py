"""
Page 3 — Recommendation Dashboard
Ranked factory reassignment suggestions across all products.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from optimizer import load_artifacts, generate_recommendations, add_risk_labels
from download_data import PRODUCT_FACTORY_MAP

st.set_page_config(page_title="Recommendations", page_icon="📋", layout="wide")
st.title("📋 Recommendation Dashboard")
st.markdown(
    "Auto-generated factory reassignment suggestions ranked by operational impact. "
    "Only moves that reduce lead time are shown."
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def get_processed():
    return pd.read_csv("data/nassau_processed.csv")

@st.cache_resource
def get_model():
    return load_artifacts()

@st.cache_data
def get_recs(_model, _encoders, _features, top_n):
    df = get_processed()
    recs = generate_recommendations(df, _model, _encoders, _features, top_n=top_n)
    return add_risk_labels(recs)

try:
    df = get_processed()
    model, encoders, feature_names = get_model()
except FileNotFoundError:
    st.error("Pipeline artifacts not found. Run the pipeline first (see Home page).")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    top_n     = st.slider("Max Recommendations", 5, 30, 15)
    min_save  = st.slider("Min Lead Time Reduction (days)", 0.0, 5.0, 0.5, 0.1)
    risk_filter = st.multiselect("Risk Level", ["Low", "Medium", "High"],
                                  default=["Low", "Medium", "High"])

recs = get_recs(model, encoders, feature_names, top_n)

if recs.empty:
    st.warning("No beneficial reassignments found with current data.")
    st.stop()

# Apply filters
recs_filtered = recs[
    (recs["Lead Time Reduction (days)"] >= min_save) &
    (recs["Risk Level"].isin(risk_filter))
].reset_index(drop=True)

# ── KPI strip ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Recommendations",   len(recs_filtered))
c2.metric("Products Covered",        recs_filtered["Product"].nunique() if not recs_filtered.empty else 0)
c3.metric("Max Saving",
          f"{recs_filtered['Lead Time Reduction (days)'].max():.1f} days" if not recs_filtered.empty else "—")
c4.metric("Avg Saving",
          f"{recs_filtered['Lead Time Reduction (days)'].mean():.1f} days" if not recs_filtered.empty else "—")

st.markdown("---")

if recs_filtered.empty:
    st.info("No recommendations match the current filters.")
    st.stop()

# ── Horizontal bar chart ───────────────────────────────────────────────────────
fig = px.bar(
    recs_filtered.head(12),
    x="Lead Time Reduction (days)",
    y="Product",
    color="Recommended Factory",
    orientation="h",
    text="Lead Time Reduction (days)",
    title="Top Reassignment Recommendations by Lead Time Savings",
    labels={"Lead Time Reduction (days)": "Days Saved"},
    height=420,
)
fig.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

# ── Scatter: savings vs profit impact ─────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    fig2 = px.scatter(
        recs_filtered,
        x="Lead Time Reduction (%)",
        y="Profit Impact Score",
        color="Risk Level",
        color_discrete_map={"Low": "#1D9E75", "Medium": "#F0B429", "High": "#E85D24"},
        size="Lead Time Reduction (days)",
        hover_data=["Product", "Current Factory", "Recommended Factory"],
        title="Savings vs Profit Impact",
        labels={
            "Lead Time Reduction (%)": "Lead Time Reduction (%)",
            "Profit Impact Score":     "Profit Impact Score",
        },
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    # Factory-level summary
    factory_summary = (
        recs_filtered.groupby("Recommended Factory")
        .agg(
            Moves=("Product", "count"),
            Avg_Reduction=("Lead Time Reduction (days)", "mean"),
            Avg_Profit=("Profit Impact Score", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("Avg_Reduction", ascending=False)
    )
    st.markdown("### By Recommended Factory")
    st.dataframe(factory_summary, use_container_width=True, hide_index=True)

# ── Full table ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Full Recommendation Table")

risk_color = {"Low": "background-color: #d4edda", "Medium": "background-color: #fff3cd",
              "High": "background-color: #f8d7da"}

def color_risk(val):
    return risk_color.get(val, "")

display_cols = [
    "Product", "Current Factory", "Recommended Factory",
    "Lead Time Reduction (days)", "Lead Time Reduction (%)",
    "Distance Delta (miles)", "Profit Impact Score", "Risk Level"
]
styled_df = recs_filtered[display_cols].style.map(color_risk, subset=["Risk Level"])
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ── Download button ────────────────────────────────────────────────────────────
csv = recs_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Recommendations CSV",
    data=csv,
    file_name="nassau_recommendations.csv",
    mime="text/csv",
)
