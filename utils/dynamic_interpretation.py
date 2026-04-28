from __future__ import annotations

import pandas as pd


def narrative_top_variables(df: pd.DataFrame, weight_col: str, label: str, top_n: int = 5) -> str:
    if df is None or df.empty or weight_col not in df.columns:
        return "No variable ranking is available yet."
    top = df.sort_values(weight_col, ascending=False).head(top_n)["feature"].tolist()
    top_text = ", ".join(str(x) for x in top)
    return (
        f"The strongest {label} signals are currently concentrated in: {top_text}. "
        f"These variables should receive the most attention when interpreting acceptance, decline, or review outcomes."
    )


def narrative_fairness(fairness_df: pd.DataFrame) -> str:
    if fairness_df is None or fairness_df.empty or fairness_df["Group"].astype(str).eq("N/A").all():
        return "No explicit sensitive proxy was available in the uploaded dataset, so formal non-discrimination testing could not be fully segmented."
    di = fairness_df["DisparateImpact"].dropna()
    eod = fairness_df["EqualOpportunityDiff"].dropna()
    worst_di = di.min() if not di.empty else None
    worst_eod = eod.abs().max() if not eod.empty else None
    if worst_di is None or worst_eod is None:
        return "Fairness diagnostics were computed, but the current segmentation is too sparse for strong conclusions."
    fairness_view = (
        "The model appears broadly balanced across observed groups."
        if worst_di >= 0.80 and worst_eod <= 0.10
        else "The fairness panel indicates some imbalance and should be reviewed before production decisions."
    )
    return (
        f"{fairness_view} Worst disparate impact is {worst_di:.3f} and the largest absolute equal opportunity difference is {worst_eod:.3f}. "
        "Use this diagnostic as a governance screen rather than a substitute for formal policy review."
    )


def narrative_forecast(final_pd: float, stage1_pd: float, baseline_pd: float) -> str:
    tier = "Low" if final_pd < 0.25 else "Moderate" if final_pd < 0.60 else "High"
    direction = "higher" if final_pd >= stage1_pd else "lower"
    return (
        f"The applicant is classified as {tier} risk with a final weighted probability of default of {final_pd:.2%}. "
        f"Relative to the Stage 1 Bayesian estimate ({stage1_pd:.2%}), the final assessment is {direction}, which indicates that model blending and credit-policy overrides are materially influencing the recommendation. "
        f"The baseline reference score is {baseline_pd:.2%}, giving analysts a transparent benchmark for override discussions while preserving non-discrimination guardrails."
    )
