from __future__ import annotations

from pathlib import Path

import streamlit as st

from auth import render_sidebar, require_login
from database import log_action
from styles import inject_css, metric_banner, render_hero
from utils.data_loader import load_data_from_upload

INTERNAL_DATA_PATH = Path(__file__).resolve().parents[1] / "data.xlsx"

st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")
inject_css()
require_login()
render_sidebar()

render_hero(
    "Data upload and governance gate",
    "Upload an Excel or CSV file, validate the POSB-style schema, inspect data quality, and establish the modelling foundation.",
    eyebrow="Data Upload",
)

load_source = st.radio("Choose data source", ["Upload file", "Load internal sample data"], horizontal=True)
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
    for key in ["bayes_result", "bayes_scored", "xgb_result", "xgb_scored"]:
        st.session_state.pop(key, None)
    log_action(st.session_state.current_user, st.session_state.user_role, log_event_name, data_source_name)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_banner("Rows", f"{len(bundle.clean):,}", "Validated portfolio observations"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_banner("Columns", f"{bundle.clean.shape[1]:,}", "Raw + engineered variables"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_banner("Schema", "Pass" if bundle.schema_ok else "Fail", "Core POSB structure validation"), unsafe_allow_html=True)

    if not bundle.schema_ok:
        st.error(f"Missing required columns: {bundle.missing_cols}")
    else:
        st.success("Schema validation passed.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Data preview")
    st.dataframe(bundle.clean.head(25), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Drift / data quality monitor")
    st.dataframe(bundle.drift_report, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Upload a dataset or load the internal sample data to start the modelling workflow.")
