"""
app.py — Nassau Candy Optimizer landing page
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Nassau Candy — Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🍬 Nassau Candy Distributor")
st.subheader("Factory Reallocation & Shipping Optimization System")
st.markdown("---")

st.markdown("""
This decision intelligence system uses machine learning to recommend optimal
factory assignments for Nassau Candy products — reducing shipping lead times
and protecting profit margins across all 5 factories and 15 products.
""")

# ── Quick-check data artifacts ─────────────────────────────────────────────────
model_ready = os.path.exists("data/best_model.pkl")
data_ready  = os.path.exists("data/nassau_processed.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Products", "15")
col2.metric("Factories", "5")
col3.metric("Ship Modes", "4")
col4.metric("Model Ready", "✅" if model_ready else "❌ Run pipeline")

if not data_ready or not model_ready:
    st.warning(
        "⚠️ Pipeline not yet run. Open a terminal and execute:\n"
        "```\n"
        "python src/download_data.py\n"
        "python src/preprocessing.py\n"
        "python src/model_train.py\n"
        "```"
    )
else:
    # Show quick model metrics
    metrics_path = "data/model_metrics.csv"
    if os.path.exists(metrics_path):
        st.markdown("### Model Performance Summary")
        metrics_df = pd.read_csv(metrics_path, index_col=0)
        st.dataframe(metrics_df.style.highlight_min(subset=["RMSE", "MAE"], color="#d4edda")
                                     .highlight_max(subset=["R2"],          color="#d4edda"),
                     use_container_width=True)

        best_name = open("data/best_model_name.txt").read().strip() if os.path.exists("data/best_model_name.txt") else "—"
        st.success(f"✅ Best model in use: **{best_name}**")

st.markdown("---")
st.markdown("""
**Navigate using the sidebar:**

- **1 Factory Optimizer** — Predict lead time for any product across all factories
- **2 What-If Analysis** — Compare current vs. proposed assignment side-by-side
- **3 Recommendations** — Ranked reassignment suggestions sorted by impact
- **4 Risk Panel** — Profit sensitivity and high-risk reassignment alerts
""")
