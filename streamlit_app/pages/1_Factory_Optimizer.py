"""
Page 1 — Factory Optimization Simulator
Select a product, destination, and shipping parameters.
See predicted lead time for every available factory.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from optimizer import load_artifacts, simulate_factory_reassignment
from download_data import PRODUCT_FACTORY_MAP
from preprocessing import STATE_CENTROIDS

st.set_page_config(page_title="Factory Optimizer", page_icon="🏭", layout="wide")
st.title("🏭 Factory Optimization Simulator")
st.markdown("Predict shipping lead time for a product if produced at any factory.")

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def get_processed():
    return pd.read_csv("data/nassau_processed.csv")

@st.cache_resource
def get_model():
    return load_artifacts()

try:
    df = get_processed()
    model, encoders, feature_names = get_model()
except FileNotFoundError:
    st.error("Pipeline artifacts not found. Run the pipeline first (see Home page).")
    st.stop()

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Inputs")
    product   = st.selectbox("Product",           sorted(PRODUCT_FACTORY_MAP.keys()))
    dest_state = st.selectbox("Destination State", sorted(STATE_CENTROIDS.keys()),
                               index=list(sorted(STATE_CENTROIDS.keys())).index("Texas"))
    ship_mode = st.selectbox("Ship Mode",          sorted(df["Ship Mode"].dropna().unique()))
    region    = st.selectbox("Region",             sorted(df["Region"].dropna().unique()))
    division  = st.selectbox("Division",           sorted(df["Division"].dropna().unique()))
    month     = st.slider("Order Month", 1, 12, 6)
    units     = st.slider("Units", 1, 500, 50)
    cost      = st.slider("Unit Cost ($)", 1, 500, 50)

quarter = (month - 1) // 3 + 1

# ── Run simulation ─────────────────────────────────────────────────────────────
results_df, current_factory = simulate_factory_reassignment(
    product, dest_state, ship_mode, region, division,
    month, quarter, units, cost,
    model, encoders, feature_names
)

# ── KPI strip ─────────────────────────────────────────────────────────────────
best_row    = results_df.iloc[0]
current_row = results_df[results_df["Is Current"]].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Factory",        current_factory)
c2.metric("Current Lead Time",      f"{current_row['Predicted Lead Time']:.1f} days")
c3.metric("Best Alternative",       best_row["Factory"] if not best_row["Is Current"] else "Current is best")
c4.metric("Max Savings",
          f"{best_row['Lead Time Reduction (days)']:.1f} days  ({best_row['Lead Time Reduction (%)']:.0f}%)"
          if not best_row["Is Current"] else "—")

st.markdown("---")

# ── Bar chart ─────────────────────────────────────────────────────────────────
color_map = {True: "#E85D24", False: "#1D9E75"}
fig = px.bar(
    results_df,
    x="Factory",
    y="Predicted Lead Time",
    color="Is Current",
    color_discrete_map=color_map,
    text="Predicted Lead Time",
    labels={"Predicted Lead Time": "Lead Time (days)", "Is Current": "Currently Assigned"},
    title=f"Predicted Lead Time by Factory — {product} → {dest_state}",
)
fig.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
fig.update_layout(showlegend=True, yaxis_title="Lead Time (days)", xaxis_title="")
st.plotly_chart(fig, use_container_width=True)

# ── Distance vs Lead Time scatter ──────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    fig2 = px.scatter(
        results_df,
        x="Distance Miles",
        y="Predicted Lead Time",
        text="Factory",
        color="Is Current",
        color_discrete_map=color_map,
        size="Profit Impact Score",
        title="Distance vs Predicted Lead Time",
        labels={"Distance Miles": "Distance (miles)", "Predicted Lead Time": "Lead Time (days)"},
    )
    fig2.update_traces(textposition="top center")
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.markdown("### Factory Comparison Table")
    display_cols = [
        "Factory", "Distance Miles", "Predicted Lead Time",
        "Lead Time Reduction (days)", "Lead Time Reduction (%)", "Profit Impact Score"
    ]
    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

# ── Recommendation box ────────────────────────────────────────────────────────
st.markdown("---")
if not best_row["Is Current"] and best_row["Lead Time Reduction (days)"] > 0:
    st.success(
        f"✅ **Recommendation:** Reassign **{product}** from **{current_factory}** "
        f"to **{best_row['Factory']}** to save **{best_row['Lead Time Reduction (days)']:.1f} days** "
        f"({best_row['Lead Time Reduction (%)']}% reduction) on orders shipping to {dest_state}."
    )
else:
    st.info(f"✅ **{current_factory}** is already the optimal factory for this route.")
