from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class BayesianResult:
    model: object
    feature_names: list[str]
    scaler: StandardScaler
    info_weights: pd.DataFrame
    posterior_summary: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    threshold: float
    posterior_covariance: np.ndarray


EXCLUDED_MODEL_INPUTS = {"MaritalStatus"}


def _drop_disallowed_inputs(X: pd.DataFrame) -> pd.DataFrame:
    id_like = [c for c in X.columns if "id" in str(c).lower() or str(c).lower().endswith("_key")]
    disallowed = [c for c in X.columns if c in EXCLUDED_MODEL_INPUTS]
    return X.drop(columns=id_like + disallowed, errors="ignore")


def _prepare_stage1_design(df: pd.DataFrame, target_col: str):
    work = df.copy()
    y = work[target_col].astype(int).to_numpy()
    X = work.drop(columns=[target_col], errors="ignore").copy()
    X = _drop_disallowed_inputs(X)
    X = pd.get_dummies(X, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Ensure a pure numeric float matrix for Streamlit Cloud / NumPy linear algebra.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(float)
    return X, Xs, y, scaler


def compute_information_weights(df: pd.DataFrame, target_col: str = "Default") -> pd.DataFrame:
    work = df.copy()
    X = work.drop(columns=[target_col], errors="ignore").copy()
    X = _drop_disallowed_inputs(X)
    X = pd.get_dummies(X, drop_first=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int)

    weights = []
    yv = y.to_numpy().astype(float)
    for col in X.columns:
        xv = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
        var = np.var(xv)
        cov = np.cov(xv, yv, ddof=0)[0, 1] if var > 0 else 0.0
        w = cov / (var + 1e-9)
        weights.append({"feature": col, "covariance": cov, "variance": var, "information_weight": w, "abs_weight": abs(w)})
    out = pd.DataFrame(weights).sort_values("abs_weight", ascending=False).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def fit_bayesian_stage(df: pd.DataFrame, target_col: str = "Default", threshold: float = 0.50, prior_precision: float = 2.0, test_size: float = 0.25, random_state: int = 42) -> BayesianResult:
    X, Xs, y, scaler = _prepare_stage1_design(df, target_col)
    info_weights = compute_information_weights(pd.concat([X, pd.Series(y, name=target_col)], axis=1), target_col=target_col)

    weights_vec = info_weights.set_index("feature").reindex(X.columns)["information_weight"].fillna(0).to_numpy(dtype=float)
    weighted_signal = X.to_numpy(dtype=float) @ weights_vec
    X_aug = np.column_stack([Xs.astype(float), weighted_signal.astype(float)]).astype(float)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_aug, y, np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state
    )

    inverse_c = 1.0 / max(prior_precision, 1e-4)
    model = LogisticRegression(C=inverse_c, solver="lbfgs", max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    beta = np.r_[model.intercept_, model.coef_.ravel()]
    X_design = np.column_stack([np.ones(X_train.shape[0]), X_train]).astype(float)
    w = (p_train * (1 - p_train)).astype(float)
    precision = X_design.T @ (X_design * w[:, None]) + float(prior_precision) * np.eye(X_design.shape[1], dtype=float)
    precision = np.asarray(precision, dtype=float)
    cov = np.linalg.pinv(precision)

    post = pd.DataFrame(
        {
            "feature": ["Intercept"] + list(X.columns) + ["Bayesian_Info_Signal"],
            "posterior_mean": beta,
            "posterior_std": np.sqrt(np.clip(np.diag(cov), 0, None)),
        }
    )
    post["lower_95"] = post["posterior_mean"] - 1.96 * post["posterior_std"]
    post["upper_95"] = post["posterior_mean"] + 1.96 * post["posterior_std"]

    train_df = df.iloc[idx_train].copy()
    test_df = df.iloc[idx_test].copy()
    train_df["PD_stage1"] = p_train
    test_df["PD_stage1"] = p_test

    return BayesianResult(
        model=model,
        feature_names=list(X.columns) + ["Bayesian_Info_Signal"],
        scaler=scaler,
        info_weights=info_weights,
        posterior_summary=post,
        train_df=train_df,
        test_df=test_df,
        threshold=threshold,
        posterior_covariance=cov,
    )


def score_bayesian_stage(result: BayesianResult, df: pd.DataFrame, target_col: str = "Default") -> pd.DataFrame:
    work = df.copy()
    X = work.drop(columns=[target_col], errors="ignore").copy()
    X = _drop_disallowed_inputs(X)
    X = pd.get_dummies(X, drop_first=True)
    for col in result.feature_names[:-1]:
        if col not in X.columns:
            X[col] = 0
    X = X[result.feature_names[:-1]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    Xs = result.scaler.transform(X).astype(float)
    weights = result.info_weights.set_index("feature").reindex(X.columns)["information_weight"].fillna(0).to_numpy(dtype=float)
    weighted_signal = X.to_numpy(dtype=float) @ weights
    X_aug = np.column_stack([Xs, weighted_signal.astype(float)]).astype(float)
    work["PD_stage1"] = result.model.predict_proba(X_aug)[:, 1]
    return work


def posterior_update_summary(old_result: BayesianResult, new_scored: pd.DataFrame) -> pd.DataFrame:
    coeff = old_result.posterior_summary[["feature", "posterior_mean", "posterior_std"]].copy()
    coeff["updated_mean_proxy"] = coeff["posterior_mean"] * 0.92 + 0.08 * coeff["posterior_mean"].mean()
    coeff["delta"] = coeff["updated_mean_proxy"] - coeff["posterior_mean"]
    return coeff
