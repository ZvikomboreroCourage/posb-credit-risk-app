from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from auth import ensure_session, login_page, render_sidebar, require_login
from database import init_db, log_action, read_recent_logs
from styles import feature_card, inject_css, metric_banner, render_hero
from utils.bayesian_engine import fit_bayesian_stage, posterior_update_summary, score_bayesian_stage
from utils.data_loader import baseline_scorecard, infer_sensitive_feature, key_risk_driver_defaults, load_data_from_upload
from utils.validation import compile_validation, population_stability_index
from utils.dynamic_interpretation import narrative_forecast, narrative_top_variables
from utils.xgb_engine import fit_xgb_stage, score_xgb_stage

st.set_page_config(
    page_title="POSB Twin-Stage Credit Studio",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
ensure_session()
inject_css()

if not st.session_state.logged_in:
    login_page()
    st.stop()

require_login()
render_sidebar()

st.markdown(
    '<div class="top-nav-wrap">', unsafe_allow_html=True
)
selected = option_menu(
    menu_title=None,
    options=[
        "Upload Engine",
        "Bayesian Analytics",
        "XGBoost Analytics",
        "Model Comparison",
        "Risk Forecasting",
    ],
    icons=["cloud-upload", "diagram-3", "lightning-charge", "bar-chart-line", "activity"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0.25rem 0 0.6rem 0", "background-color": "rgba(0,0,0,0)"},
        "icon": {"color": "#ffffff", "font-size": "16px"},
        "nav-link": {
            "font-size": "14px",
            "font-weight": "700",
            "text-align": "center",
            "margin": "0px 8px 0px 0px",
            "padding": "12px 16px",
            "border-radius": "16px",
            "background-color": "rgba(255,255,255,0.82)",
            "color": "#1f172a",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #c4b5fd 0%, #fdba74 100%)",
            "color": "#111111",
        },
    },
)
st.markdown('</div>', unsafe_allow_html=True)

bundle = st.session_state.get("bundle")
bayes_result = st.session_state.get("bayes_result")
bayes_scored = st.session_state.get("bayes_scored")
xgb_result = st.session_state.get("xgb_result")
xgb_scored = st.session_state.get("xgb_scored")

INTERNAL_DATA_PATH = Path(__file__).resolve().parent / "data.xlsx"
INTERNAL_PRIOR_PRECISION = 2.0
FIXED_PD_WEIGHTS = {"Baseline": 0.50, "Stage 1": 0.30, "Stage 2": 0.20}


def _profile_frame(df):
    cols = [c for c in ["Age","Income","LoanAmount","CreditScore","MonthsEmployed","NumCreditLines","DTIRatio"] if c in df.columns]
    profile = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        profile[c] = {"mean": float(s.mean()), "std": float(s.std(ddof=0) or 1.0)}
    return profile

def _drift_signal(old_profile, new_df):
    if not old_profile:
        return 1.0
    scores = []
    for c, stats_ in old_profile.items():
        if c in new_df.columns:
            s = pd.to_numeric(new_df[c], errors="coerce")
            old_mean = stats_["mean"]
            old_std = max(stats_["std"], 1e-6)
            scores.append(abs(float(s.mean()) - old_mean) / old_std)
    return float(sum(scores) / max(len(scores), 1))


def _decision_from_pd(pd_value: float, approve_cutoff: float, review_cutoff: float):
    if pd_value < approve_cutoff:
        return "Approve", "✅", "The applicant falls below the approval threshold and currently appears acceptable under the configured risk appetite."
    if pd_value < review_cutoff:
        return "Review", "🟠", "The applicant sits in the intermediate band and should be sent for manual review, additional documentation, or compensating-factor checks."
    return "Decline", "⛔", "The applicant exceeds the review threshold and currently falls into the decline band under the configured policy."


def _fixed_pd_weights() -> dict[str, float]:
    return FIXED_PD_WEIGHTS.copy()


def _apply_credit_policy_overrides(row: pd.Series, raw_pd: float, defaults: dict) -> tuple[float, list[str]]:
    adjusted = float(raw_pd)
    notes = []

    def _median(name: str, fallback: float) -> float:
        value = defaults.get(name, fallback)
        try:
            return max(float(value), 1e-6)
        except Exception:
            return fallback

    income_med = _median("Income", 1.0)
    loan_med = _median("LoanAmount", 1.0)
    credit_med = _median("CreditScore", 600.0)
    dti_med = _median("DTIRatio", 0.35)
    emp_med = _median("MonthsEmployed", 24.0)

    income_ratio = float(row.get("Income", income_med)) / income_med
    if income_ratio >= 1.20:
        adjusted *= 0.88
        notes.append("Higher income capacity reduced PD.")
    elif income_ratio <= 0.80:
        adjusted *= 1.12
        notes.append("Lower income capacity increased PD.")

    affordability = float(row.get("Income", income_med)) / max(float(row.get("LoanAmount", loan_med)), 1e-6)
    baseline_affordability = income_med / max(loan_med, 1e-6)
    if affordability >= baseline_affordability * 1.15:
        adjusted *= 0.92
        notes.append("Stronger repayment affordability lowered PD.")
    elif affordability <= baseline_affordability * 0.85:
        adjusted *= 1.10
        notes.append("Weaker repayment affordability raised PD.")

    score = float(row.get("CreditScore", credit_med))
    if score >= credit_med + 40:
        adjusted *= 0.90
        notes.append("Stronger character and repayment history lowered PD.")
    elif score <= credit_med - 40:
        adjusted *= 1.10
        notes.append("Weaker repayment history increased PD.")

    dti = float(row.get("DTIRatio", dti_med))
    if dti <= dti_med * 0.85:
        adjusted *= 0.93
        notes.append("Lower leverage improved capacity and reduced PD.")
    elif dti >= dti_med * 1.15:
        adjusted *= 1.10
        notes.append("Higher leverage reduced capacity and increased PD.")

    months_emp = float(row.get("MonthsEmployed", emp_med))
    if months_emp >= emp_med * 1.20:
        adjusted *= 0.95
        notes.append("Longer employment stability reduced PD.")
    elif months_emp <= emp_med * 0.80:
        adjusted *= 1.06
        notes.append("Shorter employment stability increased PD.")

    adjusted = min(max(adjusted, 0.001), 0.999)
    return adjusted, notes

if selected == "Upload Engine":
    render_hero(
        "POSB credit scoring launch engine",
        "Load the portfolio dataset, validate the schema, and prepare the Bayesian–XGBoost workflow inside a streamlined credit decision studio.",
        eyebrow="Upload Engine",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(feature_card("Schema guardrails", "Validates required POSB-style variables and auto-engineers interaction terms for model readiness."), unsafe_allow_html=True)
    with c2:
        st.markdown(feature_card("Fast training workflow", "Trains Bayesian and stacked machine learning stages only when needed, with cache acceleration."), unsafe_allow_html=True)
    with c3:
        st.markdown(feature_card("Single cockpit UX", "Uses clear horizontal navigation and high-contrast panels for a focused credit scoring workflow."), unsafe_allow_html=True)

    load_source = st.radio(
        "Choose data source",
        ["Upload file", "Load internal sample data"],
        horizontal=True,
        help="Use an uploaded client file or load the bundled sample dataset for demonstration and testing.",
    )
    uploaded = None
    load_internal = False
    if load_source == "Upload file":
        uploaded = st.file_uploader("Upload POSB dataset (Excel or CSV)", type=["xlsx", "csv"])
    else:
        load_internal = st.button("Load internal sample data", use_container_width=True)

    if uploaded or load_internal:
        if load_internal:
            if not INTERNAL_DATA_PATH.exists():
                st.error("Internal sample data file is missing. Please upload a CSV or Excel file instead.")
                st.stop()
            bundle = load_data_from_upload(INTERNAL_DATA_PATH.read_bytes(), INTERNAL_DATA_PATH.name)
            data_source_name = "internal sample data"
            log_event_name = "internal_data_loaded"
        else:
            bundle = load_data_from_upload(uploaded.getvalue(), uploaded.name)
            data_source_name = uploaded.name
            log_event_name = "data_uploaded"

        st.session_state["bundle"] = bundle
        st.session_state["current_profile"] = _profile_frame(bundle.clean)
        for key in ["bayes_result", "bayes_scored", "xgb_result", "xgb_scored", "bayes_profile", "xgb_profile"]:
            st.session_state.pop(key, None)
        log_action(st.session_state.current_user, st.session_state.user_role, log_event_name, data_source_name)

        m1, m2, m3 = st.columns(3)
        m1.markdown(metric_banner("Rows", f"{len(bundle.clean):,}", "Portfolio observations received"), unsafe_allow_html=True)
        m2.markdown(metric_banner("Columns", f"{bundle.clean.shape[1]:,}", "Raw and engineered fields"), unsafe_allow_html=True)
        m3.markdown(metric_banner("Schema", "Pass" if bundle.schema_ok else "Fail", "Required POSB structure check"), unsafe_allow_html=True)

        if bundle.schema_ok:
            st.success("Schema validation passed. The training engine is ready.")
        else:
            st.error(f"Missing required columns: {bundle.missing_cols}")

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Data preview")
        st.dataframe(bundle.clean.head(30), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Data quality and drift snapshot")
        st.dataframe(bundle.drift_report, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Recent audit activity", expanded=False):
            logs = read_recent_logs(limit=12)
            if logs:
                st.dataframe(logs, use_container_width=True, hide_index=True)
            else:
                st.info("No audit activity logged yet.")
    else:
        st.info("Upload a CSV/Excel file or load the internal sample dataset to activate the analytics engine.")

elif selected == "Bayesian Analytics":
    render_hero(
        "Stage 1 Bayesian analytics cockpit",
        "Compute covariance-based information weights, fit the prior-aware logistic layer, and inspect posterior diagnostics without leaving the main command surface.",
        eyebrow="Bayesian Analytics",
    )

    if bundle is None:
        st.info("Upload data first from the Upload Engine tab.")
        st.stop()

    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01)
    prior_precision = INTERNAL_PRIOR_PRECISION

    drift_signal = _drift_signal(st.session_state.get("bayes_profile"), bundle.clean)
    should_retrain = bayes_result is None or drift_signal > 0.25
    with st.spinner("Running Stage 1 Bayesian model..."):
        if should_retrain:
            bayes_result = fit_bayesian_stage(bundle.clean, target_col="Default", threshold=threshold, prior_precision=prior_precision)
            bayes_scored = score_bayesian_stage(bayes_result, bundle.clean)
            st.session_state["bayes_result"] = bayes_result
            st.session_state["bayes_scored"] = bayes_scored
            st.session_state["bayes_profile"] = _profile_frame(bundle.clean)
            log_action(st.session_state.current_user, st.session_state.user_role, "bayesian_model_retrained", f"threshold={threshold}, drift={drift_signal:.3f}")
        else:
            bayes_scored = score_bayesian_stage(bayes_result, bundle.clean)
            st.session_state["bayes_scored"] = bayes_scored
            log_action(st.session_state.current_user, st.session_state.user_role, "bayesian_model_reused", f"drift={drift_signal:.3f}")
        validation = compile_validation(bayes_scored, "Default", "PD_stage1", threshold, sensitive_col=infer_sensitive_feature(bundle.clean))

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_banner("AUC", f"{validation.metrics['AUC']:.3f}", "Stage 1 discrimination"), unsafe_allow_html=True)
    m2.markdown(metric_banner("KS", f"{validation.metrics['KS']:.3f}", "Good-bad separation"), unsafe_allow_html=True)
    m3.markdown(metric_banner("Recall", f"{validation.metrics['Recall']:.2%}", "Captured defaulters"), unsafe_allow_html=True)
    m4.markdown(metric_banner("Precision", f"{validation.metrics['Precision']:.2%}", "Positive call quality"), unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        fig = px.bar(
            bayes_result.info_weights.head(15),
            x="information_weight",
            y="feature",
            orientation="h",
            color="abs_weight",
            color_continuous_scale="Sunset",
            title="Bayesian information weight ranking",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(validation.roc_frame, x="fpr", y="tpr", title="ROC explorer")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns([1, 1])
    with lower_left:
        indicator = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(bayes_scored["PD_stage1"].mean() * 100),
            title={"text": "Average PD_stage1"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#f97316"},
                "steps": [
                    {"range": [0, 25], "color": "rgba(94,234,212,0.35)"},
                    {"range": [25, 60], "color": "rgba(251,191,36,0.30)"},
                    {"range": [60, 100], "color": "rgba(244,114,182,0.28)"},
                ],
            },
        ))
        st.plotly_chart(indicator, use_container_width=True)
    with lower_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("AI Analyst Commentary")
        st.write(
            f"The Bayesian stage is producing an average default probability of {bayes_scored['PD_stage1'].mean():.2%}. "
            f"The strongest evidence sits in the top-ranked information-weight variables, while posterior shrinkage is preserving coefficient stability. "
            f"At the selected threshold, the model reports AUC {validation.metrics['AUC']:.3f} and KS {validation.metrics['KS']:.3f}. "
            f"Current drift signal versus the last trained Bayesian profile is {drift_signal:.3f}, so the engine {'retrained' if should_retrain else 'reused the prior model for speed'}."
        )
        st.info(narrative_top_variables(bayes_result.info_weights.rename(columns={"abs_weight":"importance"}), "abs_weight", "Bayesian evidence"))
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Posterior summary", expanded=False):
        st.dataframe(bayes_result.posterior_summary.head(40), use_container_width=True, hide_index=True)

    with st.expander("Real-time posterior updating monitor", expanded=False):
        updated = posterior_update_summary(bayes_result, bayes_scored)
        st.dataframe(updated.head(25), use_container_width=True, hide_index=True)

elif selected == "XGBoost Analytics":
    render_hero(
        "Stage 2 XGBoost analytics cockpit",
        "Inject PD_stage1 into the stacked nonlinear learner, rebalance the class mix, and benchmark boosted performance without cluttered PD histograms.",
        eyebrow="XGBoost Analytics",
    )

    if bundle is None or bayes_scored is None:
        st.info("Run Upload Engine and Bayesian Analytics first.")
        st.stop()

    use_smote = st.toggle("Apply lightweight SMOTE-style rebalancing", value=True)

    drift_signal = _drift_signal(st.session_state.get("xgb_profile"), bayes_scored)
    should_retrain = xgb_result is None or drift_signal > 0.25
    with st.spinner("Running Stage 2 stacked model..."):
        if should_retrain:
            xgb_result = fit_xgb_stage(bayes_scored, target_col="Default", use_smote=use_smote)
            xgb_scored = score_xgb_stage(xgb_result, bayes_scored)
            st.session_state["xgb_result"] = xgb_result
            st.session_state["xgb_scored"] = xgb_scored
            st.session_state["xgb_profile"] = _profile_frame(bayes_scored)
            log_action(st.session_state.current_user, st.session_state.user_role, "xgb_model_retrained", f"use_smote={use_smote}, engine={xgb_result.engine_name}, drift={drift_signal:.3f}")
        else:
            xgb_scored = score_xgb_stage(xgb_result, bayes_scored)
            st.session_state["xgb_scored"] = xgb_scored
            log_action(st.session_state.current_user, st.session_state.user_role, "xgb_model_reused", f"engine={xgb_result.engine_name}, drift={drift_signal:.3f}")
        validation = compile_validation(xgb_scored, "Default", "PD_stage2", threshold=0.50, sensitive_col=infer_sensitive_feature(bundle.clean))

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_banner("AUC", f"{validation.metrics['AUC']:.3f}", "Stacked model AUC"), unsafe_allow_html=True)
    m2.markdown(metric_banner("KS", f"{validation.metrics['KS']:.3f}", "Separation strength"), unsafe_allow_html=True)
    m3.markdown(metric_banner("Recall", f"{validation.metrics['Recall']:.2%}", "Captured defaults"), unsafe_allow_html=True)
    m4.markdown(metric_banner("CV AUC", f"{xgb_result.cv_auc:.3f}", f"Engine: {xgb_result.engine_name}"), unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        fig = px.bar(
            xgb_result.importance_df.head(15),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Magma",
            title="Stage 2 importance ranking",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        # Replace messy histogram with clean predicted-vs-actual style scatter
        view = xgb_scored.copy()
        fig = px.scatter(
            view,
            x="PD_stage1",
            y="PD_stage2",
            color="Default",
            opacity=0.7,
            title="Stage 1 vs Stage 2 default probability surface",
            labels={"PD_stage1": "Bayesian stage PD", "PD_stage2": "Stacked stage PD"},
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(fig, use_container_width=True)

    mid_left, mid_right = st.columns([1, 1])
    with mid_left:
        # decile risk profile instead of histogram
        tmp = xgb_scored.copy()
        tmp["PD_decile"] = pd.qcut(tmp["PD_stage2"], q=10, duplicates="drop")
        dec = tmp.groupby("PD_decile", as_index=False)["Default"].mean()
        dec["PD_decile"] = dec["PD_decile"].astype(str)
        fig = px.line(dec, x="PD_decile", y="Default", markers=True, title="Observed default rate by PD decile")
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with mid_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("AI Analyst Commentary")
        st.write(
            f"The stacked engine uses PD_stage1 as a supervisory meta-signal and currently runs on {xgb_result.engine_name}. "
            f"Instead of noisy histograms, this view focuses on risk surface alignment and decile monotonicity. "
            f"Cross-validated AUC is {xgb_result.cv_auc:.3f} and current out-of-sample AUC is {validation.metrics['AUC']:.3f}. "
            f"The current drift signal versus the last trained Stage 2 profile is {drift_signal:.3f}, so the system {'retrained' if should_retrain else 'reused the pretrained model for speed'}."
        )
        st.info(narrative_top_variables(xgb_result.importance_df, "importance", "predictive"))
        st.info("Non-discrimination note: marital status is excluded from risk pricing and is not used in the Stage 2 risk rating workflow.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Model configuration", expanded=False):
        st.write(
            "These settings are selected by the training engine to balance predictive separation, stability, and execution speed. "
            "They are shown for technical review only and do not need to be adjusted during normal use."
        )
        st.json(xgb_result.best_params)

elif selected == "Model Comparison":
    render_hero(
        "Unified model comparison engine",
        "Benchmark Stage 1 Bayesian, Stage 2 XGBoost, and the POSB-style baseline scorecard using a clear management dashboard.",
        eyebrow="Model Comparison",
    )

    if bundle is None or bayes_scored is None or xgb_scored is None:
        st.info("Run the earlier tabs first.")
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

    fig = px.bar(
        comp.melt(id_vars="Model", value_vars=["AUC", "KS", "Recall", "Precision", "Specificity"]),
        x="variable", y="value", color="Model", barmode="group",
        title="Comparative performance dashboard",
        color_discrete_sequence=["#f59e0b", "#8b5cf6", "#f97316"],
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Metrics table")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        psi_stage2 = population_stability_index(bayes_scored["PD_stage1"], xgb_scored["PD_stage2"])
        best_model = comp.sort_values("AUC", ascending=False).iloc[0]["Model"]
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Adaptive recalibration monitor")
        st.write(
            "The monitor compares the distribution of Stage 1 and Stage 2 default probabilities using the Population Stability Index. "
            "It helps identify when the stacked model is producing a materially different risk profile, which can indicate that recalibration is needed before decisions rely on stale relationships."
        )
        st.write(f"Population Stability Index (PD_stage1 vs PD_stage2) = **{psi_stage2:.4f}**")
        if psi_stage2 > 0.25:
            st.error("Material drift detected. Recalibration recommended.")
        elif psi_stage2 > 0.10:
            st.warning("Moderate drift detected. Monitor closely.")
        else:
            st.success("Stable population. No urgent recalibration signal.")
        st.write(f"Current best performer by AUC: **{best_model}**.")
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Risk Forecasting":
    render_hero(
        "Live risk forecasting workbench",
        "Enter borrower characteristics to produce real-time default risk predictions from the Bayesian stage, the stacked stage, and the baseline scorecard, with policy overrides aligned to the 5Cs of credit.",
        eyebrow="Risk Forecasting",
    )

    if bundle is None or bayes_result is None or xgb_result is None:
        st.info("Train the Bayesian and XGBoost analytics first.")
        st.stop()

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Applicant risk driver input panel")
    defaults = key_risk_driver_defaults(bundle.clean)
    applicant_name = st.text_input("Applicant label", value="Potential Applicant A")
    cols = st.columns(4)
    age = cols[0].number_input("Age", min_value=18, max_value=100, value=int(defaults.get("Age", 35)))
    income = cols[1].number_input("Income", min_value=0.0, value=float(defaults.get("Income", 0.0)))
    loan_amount = cols[2].number_input("LoanAmount", min_value=0.0, value=float(defaults.get("LoanAmount", 0.0)))
    credit_score = cols[3].number_input("CreditScore", min_value=0.0, value=float(defaults.get("CreditScore", 0.0)))

    cols2 = st.columns(4)
    months_employed = cols2[0].number_input("MonthsEmployed", min_value=0.0, value=float(defaults.get("MonthsEmployed", 0.0)))
    num_credit_lines = cols2[1].number_input("NumCreditLines", min_value=0.0, value=float(defaults.get("NumCreditLines", 0.0)))
    dti_ratio = cols2[2].number_input("DTIRatio", min_value=0.0, value=float(defaults.get("DTIRatio", 0.0)))
    default_placeholder = cols2[3].selectbox("Default label placeholder", [0, 1], index=0)

    # optional categoricals if present
    row = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "DTIRatio": dti_ratio,
        "Default": default_placeholder,
    }
    for cat in bundle.categorical_cols:
        cat_l = str(cat).lower()
        if cat == "MaritalStatus" or "id" in cat_l or cat_l.endswith("_key"):
            continue
        options = sorted(bundle.clean[cat].astype(str).dropna().unique().tolist())
        if len(options) == 0:
            continue
        row[cat] = st.selectbox(cat, options, key=f"forecast_{cat}")

    st.caption("Non-discrimination guardrail: MaritalStatus is excluded from the forecast input panel and is not used in risk rating.")

    with st.expander("How the three prediction models work", expanded=False):
        st.markdown(
            """
            **Baseline model** uses a transparent scorecard built from core repayment capacity variables such as debt burden, income, credit score, employment duration, loan size, and credit exposure. It provides a simple benchmark PD.

            **Stage 1 Bayesian model** is trained on the uploaded historical portfolio after data cleaning and feature engineering. It uses Bayesian logistic logic with prior shrinkage and information weights so that the first-stage PD reflects stable portfolio evidence rather than only raw correlations.

            **Stage 2 stacked model** is trained on the same historical portfolio but also receives the Stage 1 PD as an input. This second-stage learner captures nonlinear patterns and interactions that the baseline and Bayesian layers may miss. Depending on the environment, it runs with XGBoost or a gradient-boosting fallback.

            **Final forecast PD** is not taken from only one model. The app uses a fixed blend of 50% baseline, 30% Stage 1, and 20% Stage 2. This gives the baseline the largest governance anchor, recognises that Stage 1 builds on the baseline evidence, and gives Stage 2 a smaller overlay because it builds on Stage 1 with nonlinear refinement. Policy overrides aligned to the 5Cs of credit are then applied.
            """
        )

    forecast_df = pd.DataFrame([row])
    forecast_df["Loan_to_Income"] = forecast_df["LoanAmount"] / max(forecast_df["Income"].iloc[0], 1e-6)
    forecast_df["Employment_Income_Interaction"] = forecast_df["MonthsEmployed"] * forecast_df["Income"]
    forecast_df["Score_DTI_Gap"] = forecast_df["CreditScore"] - 100 * forecast_df["DTIRatio"]
    forecast_df["Affordability_Index"] = forecast_df["Income"] / max(forecast_df["LoanAmount"].iloc[0], 1e-6)

    policy_cols = st.columns(2)
    approve_cutoff = policy_cols[0].slider("Approve cutoff", 0.01, 0.60, 0.25, 0.01, key="approve_cutoff")
    review_cutoff = policy_cols[1].slider("Review cutoff", 0.10, 0.95, 0.60, 0.01, key="review_cutoff")
    if review_cutoff <= approve_cutoff:
        review_cutoff = min(0.95, approve_cutoff + 0.05)
        st.warning("Review cutoff must be above approve cutoff. It has been adjusted automatically.")

    if st.button("Run live risk forecast", use_container_width=True):
        scored_b = score_bayesian_stage(bayes_result, forecast_df)
        scored_x = score_xgb_stage(xgb_result, scored_b)
        scored_x["PD_baseline"] = baseline_scorecard(scored_x)

        pb = float(scored_x["PD_baseline"].iloc[0])
        p1 = float(scored_x["PD_stage1"].iloc[0])
        p2 = float(scored_x["PD_stage2"].iloc[0])

        raw_pd_map = {
            "Stage 2": p2,
            "Stage 1": p1,
            "Baseline": pb,
        }
        pd_weights = _fixed_pd_weights()
        weighted_pd_raw = sum(raw_pd_map[name] * pd_weights[name] for name in raw_pd_map)
        final_pd, override_notes = _apply_credit_policy_overrides(forecast_df.iloc[0], weighted_pd_raw, defaults)

        a, b, c, d = st.columns(4)
        a.markdown(metric_banner("Baseline PD", f"{pb:.2%}", "POSB-style baseline score"), unsafe_allow_html=True)
        b.markdown(metric_banner("Stage 1 PD", f"{p1:.2%}", "Bayesian prior-aware estimate"), unsafe_allow_html=True)
        c.markdown(metric_banner("Stage 2 PD", f"{p2:.2%}", "Stacked nonlinear estimate"), unsafe_allow_html=True)
        d.markdown(metric_banner("Final Weighted PD", f"{final_pd:.2%}", "Fixed blend with policy overrides"), unsafe_allow_html=True)


        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_pd * 100,
            title={"text": "Final weighted default risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8b5cf6"},
                "steps": [
                    {"range": [0, approve_cutoff * 100], "color": "rgba(52,211,153,0.35)"},
                    {"range": [approve_cutoff * 100, review_cutoff * 100], "color": "rgba(250,204,21,0.30)"},
                    {"range": [review_cutoff * 100, 100], "color": "rgba(244,63,94,0.30)"},
                ],
            },
        ))
        st.plotly_chart(gauge, use_container_width=True)

        decision, icon, decision_text = _decision_from_pd(final_pd, approve_cutoff, review_cutoff)
        d1, d2, d3 = st.columns(3)
        d1.markdown(metric_banner("Decision", f"{icon} {decision}", "Policy recommendation"), unsafe_allow_html=True)
        d2.markdown(metric_banner("Approve < ", f"{approve_cutoff:.0%}", "Automatic approval zone"), unsafe_allow_html=True)
        d3.markdown(metric_banner("Review < ", f"{review_cutoff:.0%}", "Manual review upper bound"), unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Decision commentary")
        st.write(narrative_forecast(final_pd, p1, pb))
        st.info(decision_text)
        weight_table = pd.DataFrame(
            [{"Model": name, "PD": raw_pd_map[name], "Weight": pd_weights[name]} for name in ["Baseline", "Stage 1", "Stage 2"]]
        )
        st.dataframe(weight_table, use_container_width=True, hide_index=True)
        st.caption("PD weighting: 50% baseline, 30% Stage 1, and 20% Stage 2. Stage 1 builds on the baseline view, while Stage 2 builds on Stage 1 as a nonlinear refinement layer.")
        if override_notes:
            st.caption("Policy overrides applied: " + " ".join(override_notes))
        else:
            st.caption("No policy override adjustments were triggered beyond the model blend.")
        st.caption(f"Forecast generated for: {applicant_name}")
        st.markdown("</div>", unsafe_allow_html=True)
