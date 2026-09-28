from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk_graph.data.generator import SyntheticBundle


@dataclass(slots=True)
class CleanBundle:
    borrowers: pd.DataFrame
    employers: pd.DataFrame
    merchants: pd.DataFrame
    macro: pd.DataFrame
    loans: pd.DataFrame
    transactions: pd.DataFrame
    performance: pd.DataFrame
    quality_log: pd.DataFrame
    data_dictionary: pd.DataFrame


def clean_synthetic_bundle(bundle: SyntheticBundle) -> CleanBundle:
    borrowers = bundle.borrowers.copy()
    employers = bundle.employers
    merchants = bundle.merchants
    macro = bundle.macro
    loans = bundle.loans
    transactions = bundle.transactions
    performance = bundle.performance

    borrowers["income_missing_flag"] = 0
    missing_income_idx = borrowers.sample(frac=0.012, random_state=17).index
    borrowers.loc[missing_income_idx, "monthly_income"] = np.nan
    borrowers.loc[missing_income_idx, "income_missing_flag"] = 1
    borrowers["monthly_income"] = borrowers.groupby("employment_type")["monthly_income"].transform(lambda s: s.fillna(s.median()))

    borrowers["external_score"] = borrowers["external_score"].clip(280, 870)
    loans = loans.drop_duplicates(subset=["loan_id"]).copy()
    transactions = transactions.drop_duplicates(subset=["transaction_id"]).copy()
    performance = performance.sort_values(["loan_id", "as_of_date"]).drop_duplicates(subset=["loan_id", "as_of_date"]).copy()
    performance["has_prior_arrears"] = performance.groupby("loan_id")["current_dpd"].shift(1).fillna(0).gt(0).astype(int)
    performance["origination_month"] = performance["as_of_date"].dt.to_period("M").dt.to_timestamp()
    quality_log = pd.DataFrame(
        [
            ("borrowers", len(bundle.borrowers), len(borrowers), "income imputed by employment_type median"),
            ("loans", len(bundle.loans), len(loans), "deduplicated by loan_id"),
            ("transactions", len(bundle.transactions), len(transactions), "deduplicated by transaction_id"),
            ("performance", len(bundle.performance), len(performance), "deduplicated by loan/date and lag features added"),
        ],
        columns=["table_name", "raw_rows", "clean_rows", "transformation_note"],
    )
    return CleanBundle(
        borrowers=borrowers,
        employers=employers,
        merchants=merchants,
        macro=macro,
        loans=loans,
        transactions=transactions,
        performance=performance,
        quality_log=quality_log,
        data_dictionary=bundle.data_dictionary,
    )
