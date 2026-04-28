from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "DTIRatio",
    "Default",
]

OPTIONAL_CATEGORICALS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

BASELINE_FEATURES = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "DTIRatio"]


@dataclass
class DataBundle:
    raw: pd.DataFrame
    clean: pd.DataFrame
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_col: str
    schema_ok: bool
    missing_cols: list[str]
    drift_report: pd.DataFrame


def _read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(bio)
    return pd.read_excel(bio, engine="openpyxl")


@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes, filename: str) -> DataBundle:
    raw = _read_file(file_bytes, filename)
    return build_bundle(raw)


@st.cache_data(show_spinner=False)
def build_bundle(raw: pd.DataFrame) -> DataBundle:
    clean = raw.copy()
    clean.columns = [str(c).strip() for c in clean.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in clean.columns]
    schema_ok = len(missing) == 0

    if "Default" in clean.columns:
        clean["Default"] = pd.to_numeric(clean["Default"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    for col in clean.columns:
        if col == "LoanID":
            continue
        if clean[col].dtype == "object":
            try:
                converted = pd.to_numeric(clean[col])
                if converted.notna().mean() > 0.90:
                    clean[col] = converted
            except Exception:
                pass

    numeric_cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c]) and c != "Default" and "id" not in str(c).lower()]
    categorical_cols = [c for c in clean.columns if c not in numeric_cols + ["Default"]]

    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].fillna(clean[col].median())

    for col in categorical_cols:
        clean[col] = clean[col].astype(str).fillna("Unknown").replace({"nan": "Unknown"})

    clean["Loan_to_Income"] = clean.get("LoanAmount", 0) / np.clip(clean.get("Income", 1), 1e-6, None)
    clean["Employment_Income_Interaction"] = clean.get("MonthsEmployed", 0) * clean.get("Income", 0)
    clean["Score_DTI_Gap"] = clean.get("CreditScore", 0) - 100 * clean.get("DTIRatio", 0)
    if "LoanAmount" in clean.columns and "Income" in clean.columns:
        clean["Affordability_Index"] = clean["Income"] / np.clip(clean["LoanAmount"], 1e-6, None)

    drift_rows = []
    for col in REQUIRED_COLUMNS[:-1]:
        if col in clean.columns:
            vals = pd.to_numeric(clean[col], errors="coerce")
            drift_rows.append(
                {
                    "Feature": col,
                    "Mean": float(vals.mean()),
                    "Std": float(vals.std(ddof=0)),
                    "MissingPct": float(raw[col].isna().mean()) if col in raw.columns else 0.0,
                }
            )

    return DataBundle(
        raw=raw,
        clean=clean,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col="Default",
        schema_ok=schema_ok,
        missing_cols=missing,
        drift_report=pd.DataFrame(drift_rows),
    )


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def infer_sensitive_feature(df: pd.DataFrame) -> str | None:
    for candidate in ["Gender", "Sex", "EmploymentType", "Education", "MaritalStatus"]:
        if candidate in df.columns:
            return candidate
    return None


def baseline_scorecard(df: pd.DataFrame) -> pd.Series:
    score = (
        0.22 * (-pd.to_numeric(df["DTIRatio"], errors="coerce").fillna(0))
        + 0.18 * (pd.to_numeric(df["CreditScore"], errors="coerce").fillna(0) / 850.0)
        + 0.14 * (pd.to_numeric(df["Income"], errors="coerce").fillna(0) / max(df["Income"].median(), 1))
        + 0.12 * (pd.to_numeric(df["MonthsEmployed"], errors="coerce").fillna(0) / max(df["MonthsEmployed"].median(), 1))
        - 0.18 * (pd.to_numeric(df["LoanAmount"], errors="coerce").fillna(0) / max(df["LoanAmount"].median(), 1))
        - 0.16 * (pd.to_numeric(df["NumCreditLines"], errors="coerce").fillna(0) / max(df["NumCreditLines"].median(), 1))
    )
    return 1 / (1 + np.exp(-score))



def key_risk_driver_defaults(df: pd.DataFrame) -> dict:
    fields = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "DTIRatio"]
    out = {}
    for field in fields:
        if field in df.columns:
            out[field] = float(pd.to_numeric(df[field], errors="coerce").median())
    return out
