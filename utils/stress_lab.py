from __future__ import annotations

import numpy as np
import pandas as pd


def apply_portfolio_stress(df: pd.DataFrame, income_shock: float, loan_shock: float, score_shock: float, dti_shock: float, model: str = "stage2") -> pd.DataFrame:
    work = df.copy()
    if "Income" in work.columns:
        work["Income"] = pd.to_numeric(work["Income"], errors="coerce").fillna(0) * (1 + income_shock)
    if "LoanAmount" in work.columns:
        work["LoanAmount"] = pd.to_numeric(work["LoanAmount"], errors="coerce").fillna(0) * (1 + loan_shock)
    if "CreditScore" in work.columns:
        work["CreditScore"] = pd.to_numeric(work["CreditScore"], errors="coerce").fillna(0) + score_shock
    if "DTIRatio" in work.columns:
        work["DTIRatio"] = pd.to_numeric(work["DTIRatio"], errors="coerce").fillna(0) * (1 + dti_shock)
    work["StressTag"] = f"Income {income_shock:+.0%} | Loan {loan_shock:+.0%} | Score {score_shock:+.0f} | DTI {dti_shock:+.0%}"
    return work


def ai_credit_analyst_summary(base_pd: float, stress_pd: float, auc: float, ks: float) -> str:
    delta = stress_pd - base_pd
    direction = "higher" if delta >= 0 else "lower"
    return (
        f"The portfolio stress run indicates an average stressed probability of default of {stress_pd:.2%}, "
        f"which is {abs(delta):.2%} {direction} than the unstressed baseline. "
        f"Current discriminatory strength remains anchored by AUC {auc:.3f} and KS {ks:.3f}. "
        f"Operationally, this suggests credit appetite should tighten when debt burden and loan size rise faster than income and score quality."
    )
