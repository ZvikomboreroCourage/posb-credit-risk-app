from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auth import render_sidebar, require_login
from database import log_action
from styles import inject_css, metric_banner, render_hero
from utils.bayesian_engine import fit_bayesian_stage, posterior_update_summary, score_bayesian_stage
from utils.data_loader import infer_sensitive_feature
from utils.validation import compile_validation

st.set_page_config(page_title="Bayesian Stage", page_icon="📈", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Stage 1 Bayesian credit intelligence",
    "Compute covariance-based information weights, estimate a prior-aware logistic layer, and generate PD_stage1 with posterior diagnostics.",
    eyebrow="Bayesian Stage Analysis",
)

bundle = st.session_state.get("bundle")
if bundle is None:
    st.info("Upload data first on the Data Upload page.")
    st.stop()

threshold = st.slider("Stage 1 classification threshold", 0.05, 0.95, 0.50, 0.01)
prior_precision = 2.0

with st.spinner("Training Stage 1 Bayesian model..."):
    result = fit_bayesian_stage(bundle.clean, target_col="Default", threshold=threshold, prior_precision=prior_precision)
    full_scored = score_bayesian_stage(result, bundle.clean)
    validation = compile_validation(full_scored, "Default", "PD_stage1", threshold, sensitive_col=infer_sensitive_feature(bundle.clean))
    st.session_state["bayes_result"] = result
    st.session_state["bayes_scored"] = full_scored
    log_action(st.session_state.current_user, st.session_state.user_role, "bayesian_model_run", f"threshold={threshold}")

m1, m2, m3, m4 = st.columns(4)
m1.markdown(metric_banner("AUC", f"{validation.metrics['AUC']:.3f}", "Discriminatory power"), unsafe_allow_html=True)
m2.markdown(metric_banner("KS", f"{validation.metrics['KS']:.3f}", "Separation between good and bad accounts"), unsafe_allow_html=True)
m3.markdown(metric_banner("Recall", f"{validation.metrics['Recall']:.2%}", "Default capture rate"), unsafe_allow_html=True)
m4.markdown(metric_banner("Type II Error", f"{validation.metrics['Type_II_Error']:.2%}", "Missed defaulter risk"), unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(result.info_weights.head(15), x="information_weight", y="feature", orientation="h", color="abs_weight", title="Bayesian information weights")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.line(validation.roc_frame, x="fpr", y="tpr", title="Stage 1 ROC curve")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(full_scored["PD_stage1"].mean() * 100),
        title={"text": "Average PD_stage1"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#70e1ff"}}
    ))
    st.plotly_chart(fig, use_container_width=True)
with c4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("AI Credit Analyst Summary")
    st.write(
        f"The Bayesian stage produced an average probability of default of {full_scored['PD_stage1'].mean():.2%}. "
        f"The strongest linear evidence comes from the top-ranked information weights, while posterior shrinkage controls instability under limited data. "
        f"At the selected threshold, the model achieves AUC {validation.metrics['AUC']:.3f} and recall {validation.metrics['Recall']:.2%}."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Posterior summary")
st.dataframe(result.posterior_summary.head(30), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Real-time posterior updating panel")
updated = posterior_update_summary(result, full_scored)
st.dataframe(updated.head(20), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Fairness diagnostics")
st.dataframe(validation.fairness, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
