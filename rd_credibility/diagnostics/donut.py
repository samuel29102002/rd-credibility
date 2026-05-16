"""Donut RD: sensitivity to observations near the cutoff."""

from dataclasses import dataclass

import numpy as np

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator


@dataclass
class DonutResult:
    """
    Results from the donut RD sensitivity test.

    Attributes
    ----------
    donut_holes : list of float
        Exclusion radii around the cutoff that were tested.
    estimates : list of float
        RD estimate for each hole size.
    ses : list of float
        Standard error for each hole size.
    baseline_estimate : float
        Estimate with no exclusion (hole = 0).
    max_deviation : float
        Maximum absolute deviation from the baseline across all holes.
    """

    donut_holes: list
    estimates: list
    ses: list
    baseline_estimate: float
    max_deviation: float


class DonutRD:
    """
    Donut RD estimator for sensitivity analysis.

    Excludes observations within a radius ``delta`` of the cutoff and
    re-estimates the RD effect.  Estimates that are stable across small
    donut holes suggest robustness to localised manipulation.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    x : array_like
        Running variable.
    cutoff : float, optional
        RD threshold. Default 0.
    bandwidth : float or None, optional
        Estimation bandwidth.  If None, uses MSE-optimal. Default None.
    donut_holes : list of float or None, optional
        Exclusion radii to test.  If None, defaults to five values from
        0 to 20 % of the bandwidth. Default None.
    """

    def __init__(self, y, x, cutoff=0, bandwidth=None, donut_holes=None):
        self.y = np.asarray(y, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.bandwidth = bandwidth
        self.donut_holes = donut_holes

    def fit(self) -> DonutResult:
        """
        Estimate the RD for each donut hole size.

        Returns
        -------
        DonutResult
        """
        if self.bandwidth is None:
            h = mse_optimal_bandwidth(self.y, self.x, self.cutoff)
        else:
            h = float(self.bandwidth)

        if self.donut_holes is None:
            holes = [0.0, 0.05 * h, 0.10 * h, 0.15 * h, 0.20 * h]
        else:
            holes = [float(d) for d in self.donut_holes]

        estimates, ses = [], []
        baseline_est = None

        for delta in holes:
            mask = np.abs(self.x - self.cutoff) >= delta  # keep obs outside hole
            y_sub = self.y[mask]
            x_sub = self.x[mask]

            try:
                result = RDEstimator(
                    y_sub,
                    x_sub,
                    cutoff=self.cutoff,
                    bandwidth=h,
                ).fit()
                est, se = result.estimate, result.se
            except Exception:
                est, se = np.nan, np.nan

            if delta == 0.0 or baseline_est is None:
                if not np.isnan(est):
                    baseline_est = est

            estimates.append(float(est))
            ses.append(float(se))

        # Fallback baseline from first non-NaN estimate
        if baseline_est is None:
            valid = [e for e in estimates if not np.isnan(e)]
            baseline_est = valid[0] if valid else 0.0

        est_arr = np.asarray(estimates)
        valid_mask = ~np.isnan(est_arr)
        max_dev = float(np.max(np.abs(est_arr[valid_mask] - baseline_est))) if valid_mask.any() else 0.0

        return DonutResult(
            donut_holes=holes,
            estimates=estimates,
            ses=ses,
            baseline_estimate=float(baseline_est),
            max_deviation=max_dev,
        )
