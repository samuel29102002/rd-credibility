"""Run all RD diagnostics and cache results in st.session_state."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from rd_credibility.diagnostics.bandwidth_grid import BandwidthSensitivity
from rd_credibility.diagnostics.covariate_balance import CovariateBalance
from rd_credibility.diagnostics.mccrary import McCraryTest
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboTest
from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator
from rd_credibility.scoring.credibility import CredibilityScore


def _cache_key(df: pd.DataFrame, cutoff: float, bandwidth, poly_order: int, kernel: str) -> str:
    return f"{len(df)}_{df.columns.tolist()}_{cutoff}_{bandwidth}_{poly_order}_{kernel}"


@st.cache_data(show_spinner=False)
def run_all(
    y_vals: np.ndarray,
    x_vals: np.ndarray,
    cov_cols: list,
    cov_data: np.ndarray | None,
    cutoff: float,
    bandwidth: float | None,
    poly_order: int,
    kernel: str,
) -> dict:
    """
    Run all diagnostics and scoring. Cached by argument hash.

    Returns a dict with keys:
      rd_result, bandwidth_used, mccrary, balance, bw_grid, placebo, report
    """
    y = np.asarray(y_vals, dtype=np.float64)
    x = np.asarray(x_vals, dtype=np.float64)

    # Bandwidth
    bw = bandwidth if bandwidth is not None else mse_optimal_bandwidth(y, x, cutoff)

    # Core RD estimate
    rd_result = RDEstimator(y, x, cutoff=cutoff, bandwidth=bw,
                             poly_order=poly_order, kernel=kernel).fit()

    # McCrary test
    mccrary = McCraryTest(x, cutoff=cutoff).fit()

    # Covariate balance
    if cov_data is not None and len(cov_cols) > 0:
        covdf = pd.DataFrame(cov_data, columns=cov_cols)
        balance = CovariateBalance(x, covdf, cutoff=cutoff, bandwidth=bw).fit()
    else:
        # Placeholder with no covariates
        balance = CovariateBalance(
            x,
            pd.DataFrame({"_none": np.zeros(len(x))}),
            cutoff=cutoff,
            bandwidth=bw,
        ).fit()

    # Bandwidth grid
    bw_grid = BandwidthSensitivity(y, x, cutoff=cutoff, poly_orders=[1, 2, 3]).fit()

    # Placebo test
    placebo = PlaceboTest(y, x, cutoff=cutoff, bandwidth=bw, n_placebo=20).fit()

    # Credibility score
    report = CredibilityScore(
        mccrary_result=mccrary,
        balance_result=balance,
        bandwidth_result=bw_grid,
        placebo_result=placebo,
    ).compute()

    return {
        "rd_result": rd_result,
        "bandwidth_used": bw,
        "mccrary": mccrary,
        "balance": balance,
        "bw_grid": bw_grid,
        "placebo": placebo,
        "report": report,
    }
