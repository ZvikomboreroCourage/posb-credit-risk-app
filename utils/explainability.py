from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def compute_bayesian_explainability_matrix(scored_df: pd.DataFrame, info_weights: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    weights = info_weights.sort_values("abs_weight", ascending=False).head(top_n).copy()
    cols = [c for c in weights["feature"] if c in scored_df.columns]
    if not cols:
        return pd.DataFrame()
    z = scored_df[cols].copy()
    for col in cols:
        z[col] = (pd.to_numeric(z[col], errors="coerce").fillna(0) - pd.to_numeric(z[col], errors="coerce").fillna(0).mean()) / (pd.to_numeric(z[col], errors="coerce").fillna(0).std(ddof=0) + 1e-9)
    wmap = weights.set_index("feature")["information_weight"]
    contrib = z.mul(wmap, axis=1)
    matrix = contrib.T.reset_index().rename(columns={"index": "Feature"})
    return matrix


def shap_like_importance(df: pd.DataFrame, prob_col: str, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    base_prob = df[prob_col].mean()
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        sens = np.corrcoef(vals.fillna(vals.median()), df[prob_col])[0, 1]
        impact = abs(sens) * vals.std(ddof=0)
        rows.append({"feature": col, "impact": float(abs(impact)), "signed_sensitivity": float(sens), "base_probability": float(base_prob)})
    return pd.DataFrame(rows).sort_values("impact", ascending=False)


def feature_sensitivity_radar(df: pd.DataFrame, prob_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        corr = np.corrcoef(vals, df[prob_col])[0, 1] if vals.std(ddof=0) > 0 else 0.0
        rows.append({"feature": col, "sensitivity": float(abs(corr))})
    return pd.DataFrame(rows).sort_values("sensitivity", ascending=False).head(8)


def risk_tier_segmentation(df: pd.DataFrame, prob_col: str, n_clusters: int = 3) -> pd.DataFrame:
    work = df.copy()
    X = work[[prob_col]].fillna(work[prob_col].mean())
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    work["RiskCluster"] = model.fit_predict(X)
    cluster_mean = work.groupby("RiskCluster")[prob_col].mean().sort_values()
    mapping = {cluster: label for cluster, label in zip(cluster_mean.index, ["Low Risk", "Medium Risk", "High Risk"][:n_clusters])}
    work["RiskTier"] = work["RiskCluster"].map(mapping)
    return work
