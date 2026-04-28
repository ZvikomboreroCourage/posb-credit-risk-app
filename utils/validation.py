from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ValidationBundle:
    metrics: dict
    roc_frame: pd.DataFrame
    confusion: pd.DataFrame
    ks_frame: pd.DataFrame
    fairness: pd.DataFrame


def compute_ks(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    fpr, tpr, thr = roc_curve(y_true, prob)
    ks = np.max(tpr - fpr)
    frame = pd.DataFrame({"threshold": thr, "tpr": tpr, "fpr": fpr, "ks": tpr - fpr})
    return float(ks), frame


def classification_metrics(y_true, prob, threshold: float = 0.5) -> tuple[dict, pd.DataFrame]:
    pred = (np.asarray(prob) >= threshold).astype(int)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    ks, ks_frame = compute_ks(np.asarray(y_true), np.asarray(prob))
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "AUC": float(auc),
        "KS": float(ks),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "Type_I_Error": float(fp / max(fp + tn, 1)),
        "Type_II_Error": float(fn / max(fn + tp, 1)),
        "Specificity": float(tn / max(tn + fp, 1)),
    }
    return metrics, ks_frame


def fairness_metrics(df: pd.DataFrame, sensitive_col: str | None, target_col: str, prob_col: str, threshold: float = 0.5) -> pd.DataFrame:
    if sensitive_col is None or sensitive_col not in df.columns:
        return pd.DataFrame([{"Group": "N/A", "SelectionRate": np.nan, "TPR": np.nan, "DisparateImpact": np.nan, "EqualOpportunityDiff": np.nan}])

    work = df[[sensitive_col, target_col, prob_col]].copy()
    work["pred"] = (work[prob_col] >= threshold).astype(int)
    rows = []
    for group, grp in work.groupby(sensitive_col):
        sr = grp["pred"].mean()
        positive = grp[grp[target_col] == 1]
        tpr = positive["pred"].mean() if len(positive) else np.nan
        rows.append({"Group": group, "SelectionRate": sr, "TPR": tpr})
    out = pd.DataFrame(rows)
    ref_sr = out["SelectionRate"].max() if len(out) else np.nan
    ref_tpr = out["TPR"].max() if len(out) else np.nan
    out["DisparateImpact"] = out["SelectionRate"] / ref_sr if pd.notna(ref_sr) and ref_sr != 0 else np.nan
    out["EqualOpportunityDiff"] = out["TPR"] - ref_tpr if pd.notna(ref_tpr) else np.nan
    return out


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    q = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, q))
    if len(cuts) < 3:
        return 0.0
    exp_bin = pd.cut(expected, bins=cuts, include_lowest=True)
    act_bin = pd.cut(actual, bins=cuts, include_lowest=True)
    exp_pct = exp_bin.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    act_pct = act_bin.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    psi = ((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum()
    return float(psi)


def compile_validation(df: pd.DataFrame, target_col: str, prob_col: str, threshold: float, sensitive_col: str | None = None) -> ValidationBundle:
    metrics, ks_frame = classification_metrics(df[target_col].astype(int), df[prob_col], threshold)
    pred = (df[prob_col] >= threshold).astype(int)
    cm = confusion_matrix(df[target_col].astype(int), pred, labels=[0, 1])
    confusion = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    fpr, tpr, thr = roc_curve(df[target_col].astype(int), df[prob_col])
    roc_frame = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    fairness = fairness_metrics(df, sensitive_col, target_col, prob_col, threshold)
    return ValidationBundle(metrics=metrics, roc_frame=roc_frame, confusion=confusion, ks_frame=ks_frame, fairness=fairness)
