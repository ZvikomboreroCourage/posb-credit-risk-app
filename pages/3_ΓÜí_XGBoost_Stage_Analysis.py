from __future__ import annotations

import plotly.express as px
import streamlit as st

from auth import render_sidebar, require_login
from database import log_action
from styles import inject_css, metric_banner, render_hero
from utils.data_loader import infer_sensitive_feature
from utils.validation import compile_validation
from utils.xgb_engine import fit_xgb_stage, score_xgb_stage

st.set_page_config(page_title="XGBoost Stage", page_icon="⚡", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Stage 2 XGBoost stacked engine",
    "Fuse PD_stage1 into the second-stage feature space, rebalance the class distribution, and train the nonlinear stacked model.",
    eyebrow="XGBoost Stage Analysis",
)

bundle = st.session_state.get("bundle")
bayes_scored = st.session_state.get("bayes_scored")
if bundle is None or bayes_scored is None:
    st.info("Run Data Upload and Bayesian Stage first.")
    st.stop()

use_smote = st.toggle("Apply lightweight SMOTE-style rebalancing", value=True)

with st.spinner("Training stacked XGBoost model..."):
    result = fit_xgb_stage(bayes_scored, target_col="Default", use_smote=use_smote)
    full_scored = score_xgb_stage(result, bayes_scored)
    validation = compile_validation(full_scored, "Default", "PD_stage2", threshold=0.50, sensitive_col=infer_sensitive_feature(bundle.clean))
    st.session_state["xgb_result"] = result
    st.session_state["xgb_scored"] = full_scored
    log_action(st.session_state.current_user, st.session_state.user_role, "xgb_model_run", f"use_smote={use_smote}, engine={result.engine_name}")

m1, m2, m3, m4 = st.columns(4)
m1.markdown(metric_banner("AUC", f"{validation.metrics['AUC']:.3f}", "Stacked model AUC"), unsafe_allow_html=True)
m2.markdown(metric_banner("Precision", f"{validation.metrics['Precision']:.2%}", "Positive precision"), unsafe_allow_html=True)
m3.markdown(metric_banner("Recall", f"{validation.metrics['Recall']:.2%}", "Captured defaults"), unsafe_allow_html=True)
m4.markdown(metric_banner("CV AUC", f"{result.cv_auc:.3f}", f"Engine: {result.engine_name}"), unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(result.importance_df.head(15), x="importance", y="feature", orientation="h", title="Stage 2 feature importance")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.histogram(full_scored, x="PD_stage2", nbins=30, color="Default", title="PD_stage2 distribution by outcome")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Model configuration")
st.write("These settings are selected by the training engine to balance predictive separation, stability, and execution speed. They are shown for technical review only and do not need to be adjusted during normal use.")
st.json(result.best_params)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Fairness diagnostics")
st.dataframe(validation.fairness, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("AI Credit Analyst Summary")
st.write(
    f"The stacked model uses Bayesian stage predictions as a meta-signal and currently runs on {result.engine_name}. "
    f"Performance suggests stronger nonlinear discrimination than a purely linear score when borrower interactions matter. "
    f"Cross-validated AUC is {result.cv_auc:.3f}, while out-of-sample AUC is {validation.metrics['AUC']:.3f}."
)
st.markdown("</div>", unsafe_allow_html=True)
