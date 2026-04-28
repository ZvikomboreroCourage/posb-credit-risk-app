from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score


EXCLUDED_MODEL_INPUTS = {"MaritalStatus"}


def _drop_disallowed_inputs(X: pd.DataFrame) -> pd.DataFrame:
    id_like = [c for c in X.columns if "id" in str(c).lower() or str(c).lower().endswith("_key")]
    disallowed = [c for c in X.columns if c in EXCLUDED_MODEL_INPUTS]
    return X.drop(columns=id_like + disallowed, errors="ignore")


@dataclass
class XGBResult:
    model: object
    feature_names: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    importance_df: pd.DataFrame
    best_params: dict
    cv_auc: float
    engine_name: str


def _simple_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y
    maj = classes[np.argmax(counts)]
    minc = classes[np.argmin(counts)]
    n_to_add = counts.max() - counts.min()
    if n_to_add <= 0:
        return X, y
    X_min = X[y == minc]
    synth = X_min[rng.integers(0, len(X_min), size=n_to_add)].copy()
    noise = rng.normal(0, 0.01, size=synth.shape)
    synth = synth + noise
    y_s = np.full(n_to_add, minc)
    return np.vstack([X, synth]), np.r_[y, y_s]


def _get_xgb_model(random_state: int = 42):
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "XGBoost"
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=random_state), "GradientBoostingFallback"


def _prep_stage2(df: pd.DataFrame, target_col: str = "Default"):
    work = df.copy()
    y = work[target_col].astype(int).to_numpy()
    X = work.drop(columns=[target_col], errors="ignore").copy()
    X = _drop_disallowed_inputs(X)
    X = pd.get_dummies(X, drop_first=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


@st.cache_data(show_spinner=False)
def fit_xgb_stage(df: pd.DataFrame, target_col: str = "Default", use_smote: bool = True, n_splits: int = 4, random_state: int = 42) -> XGBResult:
    from sklearn.model_selection import train_test_split

    X, y = _prep_stage2(df, target_col=target_col)
    idx = np.arange(len(df))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.25, random_state=random_state, stratify=y
    )

    Xtr = X_train.to_numpy().astype(float)
    ytr = y_train.astype(int)
    if use_smote:
        Xtr, ytr = _simple_smote(Xtr, ytr, random_state=random_state)
    Xtr = pd.DataFrame(Xtr, columns=X_train.columns)

    model, engine_name = _get_xgb_model(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if engine_name == "XGBoost":
        grid = GridSearchCV(
            model,
            param_grid={
                "max_depth": [3, 4],
                "learning_rate": [0.05, 0.08],
                "n_estimators": [120, 180],
            },
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(Xtr, ytr)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        cv_auc = float(grid.best_score_)
    else:
        best_model = model.fit(Xtr, ytr)
        pred = best_model.predict_proba(X_train)[:, 1]
        cv_auc = float(roc_auc_score(y_train, pred))
        best_params = {"fallback": True}

    train_df = df.iloc[idx_train].copy()
    test_df = df.iloc[idx_test].copy()
    train_df["PD_stage2"] = best_model.predict_proba(X_train)[:, 1]
    test_df["PD_stage2"] = best_model.predict_proba(X_test)[:, 1]

    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importance = np.abs(best_model.coef_).ravel()
    else:
        importance = np.zeros(X.shape[1])

    imp_df = pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values("importance", ascending=False)

    return XGBResult(
        model=best_model,
        feature_names=list(X.columns),
        train_df=train_df,
        test_df=test_df,
        importance_df=imp_df.reset_index(drop=True),
        best_params=best_params,
        cv_auc=cv_auc,
        engine_name=engine_name,
    )


def score_xgb_stage(result: XGBResult, df: pd.DataFrame, target_col: str = "Default") -> pd.DataFrame:
    work = df.copy()
    X = work.drop(columns=[target_col], errors="ignore").copy()
    X = _drop_disallowed_inputs(X)
    X = pd.get_dummies(X, drop_first=True)
    for col in result.feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[result.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["PD_stage2"] = result.model.predict_proba(X)[:, 1]
    return work
