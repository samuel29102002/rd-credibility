"""Covariate balance test for RD designs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rd_credibility.estimation.rdrobust import RDEstimator


@dataclass
class CovariateBalanceResult:
    """
    Results from the covariate balance test.

    Attributes
    ----------
    results : pd.DataFrame
        Per-covariate results with columns: covariate, estimate, se,
        p_value, significant.
    n_significant : int
        Number of covariates with p_value < 0.05.
    overall_conclusion : str
        Aggregate assessment of covariate balance.
    mean_p_value : float
        Average p-value across all covariates.
    """

    results: pd.DataFrame
    n_significant: int
    overall_conclusion: str
    mean_p_value: float


class CovariateBalance:
    """
    Test for pre-treatment covariate balance at the RD cutoff.

    For each covariate, runs an RD regression with that covariate as
    the outcome.  Under local randomisation at the cutoff, no covariate
    should exhibit a discontinuity.

    Parameters
    ----------
    x : array_like
        Running variable.
    covariates : pd.DataFrame
        Pre-treatment covariates; each column is one covariate.
    cutoff : float, optional
        RD threshold. Default 0.
    bandwidth : float or None, optional
        Common bandwidth. If None, the MSE-optimal bandwidth is
        computed separately for each covariate. Default None.
    """

    def __init__(self, x, covariates, cutoff=0, bandwidth=None):
        self.x = np.asarray(x, dtype=np.float64)
        self.covariates = (
            covariates
            if isinstance(covariates, pd.DataFrame)
            else pd.DataFrame(covariates)
        )
        self.cutoff = float(cutoff)
        self.bandwidth = bandwidth

    def fit(self) -> CovariateBalanceResult:
        """
        Run the covariate balance test for all covariates.

        Returns
        -------
        CovariateBalanceResult
        """
        rows = []
        for col in self.covariates.columns:
            z = self.covariates[col].values.astype(np.float64)
            try:
                result = RDEstimator(
                    z,
                    self.x,
                    cutoff=self.cutoff,
                    bandwidth=self.bandwidth,
                ).fit()
                rows.append(
                    {
                        "covariate": col,
                        "estimate": result.estimate,
                        "se": result.se,
                        "p_value": result.p_value,
                        "significant": result.p_value < 0.05,
                    }
                )
            except Exception:
                rows.append(
                    {
                        "covariate": col,
                        "estimate": np.nan,
                        "se": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                    }
                )

        df = pd.DataFrame(rows)
        n_sig = int(df["significant"].sum())
        mean_p = float(df["p_value"].mean(skipna=True))

        # Flag imbalance when more than 10 % of covariates (or at least 2)
        # are significant
        if n_sig > max(1, int(np.ceil(0.1 * len(df)))):
            conclusion = "Potential covariate imbalance detected"
        else:
            conclusion = "Covariates appear balanced"

        return CovariateBalanceResult(
            results=df,
            n_significant=n_sig,
            overall_conclusion=conclusion,
            mean_p_value=mean_p,
        )
