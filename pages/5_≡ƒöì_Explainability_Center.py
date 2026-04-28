from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auth import render_sidebar, require_login
from styles import inject_css, render_hero
from utils.explainability import (
    compute_bayesian_explainability_matrix,
    feature_sensitivity_radar,
    risk_tier_segmentation,
    shap_like_importance,
)

st.set_page_config(page_title="Explainability Center", page_icon="🔍", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Explainability center",
    "Open the Bayesian Explainability Matrix, sensitivity radar, SHAP-like importance, and post-model risk clustering for transparent decision support.",
    eyebrow="Explainability",
)

bundle = st.session_state.get("bundle")
bayes_result = st.session_state.get("bayes_result")
xgb_scored = st.session_state.get("xgb_scored")
if bundle is None or bayes_result is None or xgb_scored is None:
    st.info("Run Bayesian and XGBoost stages first.")
    st.stop()

matrix = compute_bayesian_explainability_matrix(xgb_scored, bayes_result.info_weights, top_n=10)
shap_df = shap_like_importance(xgb_scored, "PD_stage2", [c for c in bundle.numeric_cols if c in xgb_scored.columns][:12])
radar_df = feature_sensitivity_radar(xgb_scored, "PD_stage2", [c for c in bundle.numeric_cols if c in xgb_scored.columns][:10])
clustered = risk_tier_segmentation(xgb_scored, "PD_stage2", n_clusters=3)
st.session_state["clustered_scored"] = clustered

c1, c2 = st.columns(2)
with c1:
    if not shap_df.empty:
        fig = px.bar(shap_df.head(12), x="impact", y="feature", orientation="h", color="signed_sensitivity", title="SHAP-like importance proxy")
        st.plotly_chart(fig, use_container_width=True)
with c2:
    if not radar_df.empty:
        fig = go.Figure(go.Scatterpolar(r=radar_df["sensitivity"], theta=radar_df["feature"], fill="toself"))
        fig.update_layout(title="Feature sensitivity radar", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Bayesian Explainability Matrix")
if not matrix.empty:
    display = matrix.iloc[:, : min(16, matrix.shape[1])]
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.info("Explainability matrix could not be built from the current feature set.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Risk tier clustering")
st.dataframe(
    clustered[["PD_stage2", "RiskCluster", "RiskTier"] + ([bundle.target_col] if bundle.target_col in clustered.columns else [])].head(30),
    use_container_width=True,
    hide_index=True,
)
fig = px.histogram(clustered, x="PD_stage2", color="RiskTier", nbins=30, title="Risk tier segmentation distribution")
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
