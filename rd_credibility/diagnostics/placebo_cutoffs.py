"""Placebo cutoff test for RD design validity."""

from dataclasses import dataclass

import numpy as np

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator


@dataclass
class PlaceboResult:
    """
    Results from the placebo cutoff test.

    Attributes
    ----------
    placebo_cutoffs : np.ndarray
        Placebo threshold values used.
    placebo_estimates : np.ndarray
        RD estimate at each placebo cutoff.
    placebo_ses : np.ndarray
        Standard error at each placebo cutoff.
    true_estimate : float
        RD estimate at the true cutoff.
    true_se : float
        Standard error at the true cutoff.
    n_significant_placebos : int
        Number of placebo estimates with p_value < 0.05.
    conclusion : str
        Interpretation of the placebo test.
    """

    placebo_cutoffs: np.ndarray
    placebo_estimates: np.ndarray
    placebo_ses: np.ndarray
    true_estimate: float
    true_se: float
    n_significant_placebos: int
    conclusion: str


class PlaceboTest:
    """
    Placebo cutoff test for RD design validity.

    Estimates the RD effect at a grid of artificial cutoffs where no
    treatment is assigned.  Under a valid design these estimates should
    be centred on zero and mostly insignificant.  The true estimate is
    re-estimated using the same bandwidth for comparability.

    Placebo cutoffs are generated symmetrically to the left and right
    of the true cutoff, avoiding boundaries and the cutoff itself.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    x : array_like
        Running variable.
    cutoff : float, optional
        True RD threshold. Default 0.
    bandwidth : float or None, optional
        Common bandwidth for all estimates.  If None, the MSE-optimal
        bandwidth is computed from the true cutoff data. Default None.
    n_placebo : int, optional
        Total number of placebo cutoffs (split evenly left/right). Default 20.
    poly_order : int, optional
        Local polynomial order. Default 1.
    """

    def __init__(self, y, x, cutoff=0, bandwidth=None, n_placebo=20, poly_order=1):
        self.y = np.asarray(y, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.bandwidth = bandwidth
        self.n_placebo = int(n_placebo)
        self.poly_order = int(poly_order)

    def fit(self) -> PlaceboResult:
        """
        Run the true RD and all placebo RDs.

        Returns
        -------
        PlaceboResult
        """
        # Shared bandwidth
        if self.bandwidth is None:
            h = mse_optimal_bandwidth(self.y, self.x, self.cutoff)
        else:
            h = float(self.bandwidth)

        # True estimate
        true_res = RDEstimator(
            self.y,
            self.x,
            cutoff=self.cutoff,
            bandwidth=h,
            poly_order=self.poly_order,
        ).fit()

        # Placebo cutoffs: each placebo needs h of data on each side, so buffer
        # by h from the data extremes. The separation from the true cutoff
        # must be >= h so that no placebo bandwidth window straddles the true
        # discontinuity (which would create false positive placebo effects).
        x_min, x_max = self.x.min(), self.x.max()
        cutoff_sep = h

        n_left = self.n_placebo // 2
        n_right = self.n_placebo - n_left

        left_lo = x_min + h
        left_hi = self.cutoff - cutoff_sep
        right_lo = self.cutoff + cutoff_sep
        right_hi = x_max - h

        parts = []
        if left_hi > left_lo:
            parts.append(np.linspace(left_lo, left_hi, n_left))
        if right_hi > right_lo:
            parts.append(np.linspace(right_lo, right_hi, n_right))

        if not parts:
            return PlaceboResult(
                placebo_cutoffs=np.array([]),
                placebo_estimates=np.array([]),
                placebo_ses=np.array([]),
                true_estimate=float(true_res.estimate),
                true_se=float(true_res.se),
                n_significant_placebos=0,
                conclusion="Insufficient data range for placebo tests",
            )

        placebo_cutoffs = np.concatenate(parts)
        estimates, ses, p_values = [], [], []

        for c_star in placebo_cutoffs:
            try:
                res = RDEstimator(
                    self.y,
                    self.x,
                    cutoff=float(c_star),
                    bandwidth=h,
                    poly_order=self.poly_order,
                ).fit()
                estimates.append(res.estimate)
                ses.append(res.se)
                p_values.append(res.p_value)
            except Exception:
                estimates.append(np.nan)
                ses.append(np.nan)
                p_values.append(np.nan)

        p_arr = np.asarray(p_values)
        valid = ~np.isnan(p_arr)
        n_sig = int(np.sum(p_arr[valid] < 0.05))
        n_valid = int(valid.sum())

        sig_frac = n_sig / n_valid if n_valid > 0 else 0.0
        if sig_frac > 0.10:
            conclusion = (
                f"{n_sig}/{n_valid} placebo estimates significant "
                "— potential design concern"
            )
        else:
            conclusion = (
                f"{n_sig}/{n_valid} placebo estimates significant "
                "— no evidence of systematic bias"
            )

        return PlaceboResult(
            placebo_cutoffs=placebo_cutoffs,
            placebo_estimates=np.asarray(estimates),
            placebo_ses=np.asarray(ses),
            true_estimate=float(true_res.estimate),
            true_se=float(true_res.se),
            n_significant_placebos=n_sig,
            conclusion=conclusion,
        )
