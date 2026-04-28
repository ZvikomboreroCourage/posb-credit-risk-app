from __future__ import annotations

import plotly.express as px
import streamlit as st

from auth import render_sidebar, require_login
from styles import inject_css, render_hero
from utils.bayesian_engine import score_bayesian_stage
from utils.stress_lab import ai_credit_analyst_summary, apply_portfolio_stress
from utils.xgb_engine import score_xgb_stage
from utils.validation import compile_validation

st.set_page_config(page_title="Stress Test Lab", page_icon="🧪", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Portfolio-wide stress testing lab",
    "Shock affordability, leverage, and score quality to simulate how portfolio PD migrates under stress.",
    eyebrow="Stress Lab",
)

bundle = st.session_state.get("bundle")
bayes_result = st.session_state.get("bayes_result")
xgb_result = st.session_state.get("xgb_result")
xgb_scored = st.session_state.get("xgb_scored")
if bundle is None or bayes_result is None or xgb_result is None or xgb_scored is None:
    st.info("Run the modelling stages first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
income_shock = c1.slider("Income shock", -0.50, 0.50, -0.10, 0.01)
loan_shock = c2.slider("Loan amount shock", -0.50, 0.50, 0.15, 0.01)
score_shock = c3.slider("Credit score shock", -200, 100, -40, 5)
dti_shock = c4.slider("DTI shock", -0.50, 0.80, 0.20, 0.01)

stressed = apply_portfolio_stress(bundle.clean, income_shock, loan_shock, score_shock, dti_shock)
stressed = score_bayesian_stage(bayes_result, stressed)
stressed = score_xgb_stage(xgb_result, stressed)
validation = compile_validation(stressed, "Default", "PD_stage2", 0.50)

base_pd = xgb_scored["PD_stage2"].mean()
stress_pd = stressed["PD_stage2"].mean()

fig = px.histogram(
    stressed.assign(State="Stress").rename(columns={"PD_stage2": "PD"}),
    x="PD", nbins=30, title="Stressed PD distribution", color="Default"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("AI Credit Analyst Summary")
st.write(ai_credit_analyst_summary(base_pd, stress_pd, validation.metrics["AUC"], validation.metrics["KS"]))
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Stress results table")
show = stressed[["Age", "Income", "LoanAmount", "CreditScore", "DTIRatio", "PD_stage1", "PD_stage2", "StressTag", "Default"]].head(40)
st.dataframe(show, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
