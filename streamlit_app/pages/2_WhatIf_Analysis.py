"""
Page 2 — What-If Scenario Analysis
Compare current factory assignment vs a proposed alternative side by side.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from optimizer import load_artifacts, predict_lead_time
from download_data import PRODUCT_FACTORY_MAP, FACTORIES
from preprocessing import STATE_CENTROIDS
from geopy.distance import geodesic

st.set_page_config(page_title="What-If Analysis", page_icon="🔄", layout="wide")
st.title("🔄 What-If Scenario Analysis")
st.markdown("Compare the **current** factory assignment against a **proposed** alternative.")

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

US_CENTER = (39.5, -98.35)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Inputs")
    product       = st.selectbox("Product", sorted(PRODUCT_FACTORY_MAP.keys()))
    proposed_fac  = st.selectbox("Proposed Factory", sorted(FACTORIES.keys()))
    dest_state    = st.selectbox("Destination State", sorted(STATE_CENTROIDS.keys()),
                                  index=list(sorted(STATE_CENTROIDS.keys())).index("Texas"))
    ship_mode     = st.selectbox("Ship Mode", sorted(df["Ship Mode"].dropna().unique()))
    region        = st.selectbox("Region",    sorted(df["Region"].dropna().unique()))
    division      = st.selectbox("Division",  sorted(df["Division"].dropna().unique()))
    month         = st.slider("Order Month", 1, 12, 6)
    units         = st.slider("Units", 1, 500, 50)
    cost          = st.slider("Unit Cost ($)", 1, 500, 50)
    priority      = st.slider("Optimisation Priority", 0, 100, 50,
                               help="0 = maximise profit, 100 = minimise lead time")

quarter = (month - 1) // 3 + 1
current_factory = PRODUCT_FACTORY_MAP[product]
dest_coords     = STATE_CENTROIDS.get(dest_state, US_CENTER)

# ── Compute both scenarios ─────────────────────────────────────────────────────
current_dist  = geodesic(FACTORIES[current_factory], dest_coords).miles
proposed_dist = geodesic(FACTORIES[proposed_fac],    dest_coords).miles

current_lt  = predict_lead_time(model, encoders, feature_names,
                                 current_factory,  ship_mode, region, division,
                                 current_dist,  month, quarter, units, cost)
proposed_lt = predict_lead_time(model, encoders, feature_names,
                                 proposed_fac,     ship_mode, region, division,
                                 proposed_dist, month, quarter, units, cost)

reduction_days = current_lt - proposed_lt
reduction_pct  = reduction_days / max(current_lt, 0.01) * 100

# Profit impact proxy: distance-based cost uplift ±30%
current_profit_idx  = 1.0
proposed_profit_idx = 1 - (proposed_dist - current_dist) / max(proposed_dist, 1) * 0.30

# Weighted score combining lead time and profit
weight_lt     = priority / 100
weight_profit = 1 - weight_lt
composite_score = round(weight_lt * reduction_pct / 100 + weight_profit * proposed_profit_idx, 3)

# ── KPI strip ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Lead Time",  f"{current_lt:.1f} days")
c2.metric("Proposed Lead Time", f"{proposed_lt:.1f} days",
          delta=f"{reduction_days:+.1f} days", delta_color="inverse")
c3.metric("Distance Change",
          f"{proposed_dist - current_dist:+.0f} mi",
          delta_color="off")
c4.metric("Composite Score",    f"{composite_score:.3f}")

st.markdown("---")

# ── Side-by-side gauge charts ──────────────────────────────────────────────────
col_l, col_r = st.columns(2)

def make_gauge(value, title, max_val=15, color="#E85D24"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16}},
        number={"suffix": " days", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, max_val * 0.33],  "color": "#d4edda"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#fff3cd"},
                {"range": [max_val * 0.66, max_val],         "color": "#f8d7da"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=0, l=30, r=30))
    return fig

max_val = max(current_lt, proposed_lt, 7) * 1.2
col_l.plotly_chart(make_gauge(current_lt,  f"Current: {current_factory}",  max_val, "#E85D24"), use_container_width=True)
col_r.plotly_chart(make_gauge(proposed_lt, f"Proposed: {proposed_fac}", max_val, "#1D9E75"),   use_container_width=True)

# ── Waterfall comparison ───────────────────────────────────────────────────────
fig_wf = go.Figure(go.Waterfall(
    name="Lead Time",
    orientation="v",
    measure=["absolute", "relative", "total"],
    x=["Current Lead Time", "Change", "Proposed Lead Time"],
    y=[current_lt, -reduction_days, proposed_lt],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    decreasing={"marker": {"color": "#1D9E75"}},
    increasing={"marker": {"color": "#E85D24"}},
    totals={"marker": {"color": "#378ADD"}},
    text=[f"{current_lt:.1f}d", f"{reduction_days:+.1f}d", f"{proposed_lt:.1f}d"],
    textposition="outside",
))
fig_wf.update_layout(
    title="Lead Time Waterfall: Current → Proposed",
    yaxis_title="Days",
    showlegend=False,
    height=380,
)
st.plotly_chart(fig_wf, use_container_width=True)

# ── Summary table ──────────────────────────────────────────────────────────────
st.markdown("### Scenario Summary")
summary = pd.DataFrame({
    "Metric": [
        "Factory", "Distance (miles)", "Predicted Lead Time (days)",
        "Lead Time Reduction (days)", "Lead Time Reduction (%)", "Profit Impact Index"
    ],
    "Current": [
        current_factory, f"{current_dist:.0f}", f"{current_lt:.2f}",
        "—", "—", f"{current_profit_idx:.3f}"
    ],
    "Proposed": [
        proposed_fac, f"{proposed_dist:.0f}", f"{proposed_lt:.2f}",
        f"{reduction_days:.2f}", f"{reduction_pct:.1f}%", f"{proposed_profit_idx:.3f}"
    ],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Verdict ────────────────────────────────────────────────────────────────────
st.markdown("---")
if reduction_days > 0.5:
    st.success(
        f"✅ **Switch recommended.** Moving **{product}** from **{current_factory}** to "
        f"**{proposed_fac}** saves **{reduction_days:.1f} days** ({reduction_pct:.0f}%)."
    )
elif reduction_days < -0.5:
    st.error(
        f"❌ **Switch not recommended.** Proposed factory adds **{abs(reduction_days):.1f} days** "
        f"of lead time. Retain **{current_factory}**."
    )
else:
    st.info("ℹ️ Negligible difference between factories for this route.")
