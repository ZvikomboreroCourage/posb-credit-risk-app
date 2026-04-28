from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from auth import render_sidebar, require_login
from styles import inject_css, render_hero
from utils.data_loader import baseline_scorecard, infer_sensitive_feature
from utils.validation import compile_validation, population_stability_index

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Model comparison engine",
    "Benchmark Stage 1 Bayesian, Stage 2 XGBoost, and the POSB-style baseline scorecard with a unified validation suite.",
    eyebrow="Model Comparison",
)

bundle = st.session_state.get("bundle")
bayes_scored = st.session_state.get("bayes_scored")
xgb_scored = st.session_state.get("xgb_scored")
if bundle is None or bayes_scored is None or xgb_scored is None:
    st.info("Run the previous pages first.")
    st.stop()

baseline_df = bundle.clean.copy()
baseline_df["PD_baseline"] = baseline_scorecard(baseline_df)
sensitive = infer_sensitive_feature(bundle.clean)

val_base = compile_validation(baseline_df, "Default", "PD_baseline", 0.50, sensitive)
val_b1 = compile_validation(bayes_scored, "Default", "PD_stage1", 0.50, sensitive)
val_b2 = compile_validation(xgb_scored, "Default", "PD_stage2", 0.50, sensitive)

comp = pd.DataFrame(
    [
        {"Model": "POSB Baseline", **val_base.metrics},
        {"Model": "Stage 1 Bayesian", **val_b1.metrics},
        {"Model": "Stage 2 XGBoost", **val_b2.metrics},
    ]
)

fig = px.bar(comp.melt(id_vars="Model", value_vars=["AUC", "KS", "Recall", "Precision", "Specificity"]),
             x="variable", y="value", color="Model", barmode="group", title="Comparative performance dashboard")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Metrics table")
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    psi_stage2 = population_stability_index(bayes_scored["PD_stage1"], xgb_scored["PD_stage2"])
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Adaptive recalibration signal")
    st.write("The monitor compares the distribution of Stage 1 and Stage 2 default probabilities using the Population Stability Index. It highlights material movement in risk profiles so stale relationships can be recalibrated before they affect credit decisions.")
    st.write(f"Population Stability Index (PD_stage1 vs PD_stage2) = **{psi_stage2:.4f}**")
    if psi_stage2 > 0.25:
        st.error("Material drift detected. Recalibration recommended.")
    elif psi_stage2 > 0.10:
        st.warning("Moderate drift detected. Monitor closely.")
    else:
        st.success("Stable population. No urgent recalibration signal.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Narrative insight")
best_model = comp.sort_values("AUC", ascending=False).iloc[0]["Model"]
st.write(
    f"Across the selected metrics, **{best_model}** currently delivers the strongest discriminatory power. "
    f"The comparison engine is designed to help management understand not only overall AUC but also operational trade-offs between recall, precision, and error asymmetry."
)
st.markdown("</div>", unsafe_allow_html=True)
