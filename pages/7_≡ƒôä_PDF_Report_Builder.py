from __future__ import annotations

import io
from datetime import datetime

import plotly.express as px
import streamlit as st

from auth import render_sidebar, require_login
from database import log_action
from styles import inject_css, render_hero
from utils.data_loader import infer_sensitive_feature
from utils.validation import compile_validation

st.set_page_config(page_title="PDF Report Builder", page_icon="📄", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Regulator-ready PDF report builder",
    "Generate a concise scoring report summarising model performance, fairness diagnostics, and portfolio risk posture.",
    eyebrow="PDF Reporting",
)

if st.session_state.get("user_role") not in {"Manager", "Admin"}:
    st.error("Only Manager and Admin users can generate regulator-ready reports.")
    st.stop()

bundle = st.session_state.get("bundle")
bayes_scored = st.session_state.get("bayes_scored")
xgb_scored = st.session_state.get("xgb_scored")

if bundle is None or bayes_scored is None or xgb_scored is None:
    st.info("Run the modelling workflow first.")
    st.stop()

sensitive = infer_sensitive_feature(bundle.clean)
stage1_val = compile_validation(bayes_scored, "Default", "PD_stage1", 0.50, sensitive)
stage2_val = compile_validation(xgb_scored, "Default", "PD_stage2", 0.50, sensitive)

st.write("Preview the metrics before exporting.")
st.dataframe(
    [
        {"Model": "Stage 1 Bayesian", **stage1_val.metrics},
        {"Model": "Stage 2 XGBoost", **stage2_val.metrics},
    ],
    use_container_width=True,
    hide_index=True,
)

def build_pdf() -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=32, leftMargin=34, rightMargin=34)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("POSB Bayesian–XGBoost Credit Scoring Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Executive summary", styles["Heading2"]))
    summary = (
        f"Stage 1 Bayesian AUC = {stage1_val.metrics['AUC']:.3f}, "
        f"Stage 2 XGBoost AUC = {stage2_val.metrics['AUC']:.3f}. "
        f"Fairness diagnostics were computed using the available sensitive proxy: {sensitive or 'None'}."
    )
    story.append(Paragraph(summary, styles["BodyText"]))
    story.append(Spacer(1, 12))

    data = [["Metric", "Bayesian", "XGBoost"]]
    keys = ["AUC", "KS", "Recall", "Precision", "Specificity", "Type_I_Error", "Type_II_Error"]
    for k in keys:
        data.append([k, f"{stage1_val.metrics[k]:.4f}", f"{stage2_val.metrics[k]:.4f}"])

    table = Table(data, colWidths=[150, 120, 120])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d2d52")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9bb8d5")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f9ff")),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Governance note", styles["Heading2"]))
    story.append(Paragraph("This report is intended to support internal model governance, portfolio monitoring, and regulator-facing discussion.", styles["BodyText"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

pdf_bytes = build_pdf()
st.download_button(
    "Download PDF scoring report",
    pdf_bytes,
    file_name="posb_credit_scoring_report.pdf",
    mime="application/pdf",
)
log_action(st.session_state.current_user, st.session_state.user_role, "pdf_report_generated", "Regulator-ready report generated")
