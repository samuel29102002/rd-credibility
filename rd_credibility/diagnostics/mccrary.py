"""McCrary (2008) density discontinuity test for RD validity."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from rd_credibility.estimation.kernels import triangular


@dataclass
class McCraryResult:
    """
    Results from the McCrary (2008) density discontinuity test.

    Attributes
    ----------
    theta : float
        Log difference in estimated densities at the cutoff:
        log(f_right) - log(f_left).
    se : float
        Standard error of theta via the delta method.
    t_stat : float
        Test statistic theta / se.
    p_value : float
        Two-sided p-value under the standard normal.
    bin_centers : np.ndarray
        Centers of histogram bins.
    bin_counts : np.ndarray
        Raw observation counts per bin.
    fitted_left : np.ndarray
        Local linear fitted densities at left-side bin centers.
    fitted_right : np.ndarray
        Local linear fitted densities at right-side bin centers.
    conclusion : str
        Human-readable test verdict.
    """

    theta: float
    se: float
    t_stat: float
    p_value: float
    bin_centers: np.ndarray
    bin_counts: np.ndarray
    fitted_left: np.ndarray
    fitted_right: np.ndarray
    conclusion: str


def _wls_local_linear(bin_centers, bin_densities, cutoff, bandwidth):
    """
    Fit WLS local linear to binned densities using a triangular kernel.

    Parameters
    ----------
    bin_centers : np.ndarray
    bin_densities : np.ndarray
    cutoff : float
    bandwidth : float

    Returns
    -------
    intercept : float
        Fitted density at x = cutoff.
    var_intercept : float
        HC1 variance of the intercept.
    fitted : np.ndarray
        Fitted values at all bin_centers.
    """
    xc = bin_centers - cutoff
    y = bin_densities
    w = triangular(xc / bandwidth)

    # If bandwidth excludes almost all bins, use uniform weights
    if (w > 0).sum() < 3:
        w = np.ones_like(xc)

    X = np.column_stack([np.ones_like(xc), xc])
    Xw = X * w[:, None]        # row i: w_i * [1, xc_i]

    XtWX = Xw.T @ X            # = X' diag(w) X
    XtWy = Xw.T @ y            # = X' diag(w) y

    try:
        XtWX_inv = np.linalg.inv(XtWX)
        coeffs = XtWX_inv @ XtWy
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
        XtWX_inv = np.linalg.pinv(XtWX)

    residuals = y - X @ coeffs

    # HC1 sandwich: meat = sum_i w_i^2 e_i^2 x_i x_i'
    Xwe = Xw * residuals[:, None]
    meat = Xwe.T @ Xwe
    n, p = X.shape
    V = XtWX_inv @ meat @ XtWX_inv * (n / max(n - p, 1))

    return float(coeffs[0]), float(V[0, 0]), X @ coeffs


class McCraryTest:
    """
    McCrary (2008) running variable density discontinuity test.

    Tests H0: the density of the running variable is continuous at the
    cutoff.  Rejection suggests strategic sorting (manipulation).

    The algorithm:
    1. Bin the running variable with bins aligned to the cutoff.
    2. Normalise bin counts to density estimates.
    3. Fit local linear regression (triangular kernel) on each side.
    4. Compute theta = log(f_hat_right) - log(f_hat_left) and its SE.

    Parameters
    ----------
    x : array_like
        Running variable.
    cutoff : float, optional
        RD threshold. Default 0.
    binwidth : float or None, optional
        Histogram bin width. Defaults to the Freedman-Diaconis rule.
    bandwidth : float or None, optional
        Smoothing bandwidth for local linear on binned densities.
        Defaults to max(5 * binwidth, Silverman n^{-1/5} * std).

    References
    ----------
    McCrary, J. (2008). Manipulation of the running variable in the
    regression discontinuity design: A density test. Journal of
    Econometrics, 142(2), 698-714.
    """

    def __init__(self, x, cutoff=0, binwidth=None, bandwidth=None):
        self.x = np.asarray(x, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.binwidth = binwidth
        self.bandwidth = bandwidth

    def fit(self) -> McCraryResult:
        """
        Run the density discontinuity test.

        Returns
        -------
        McCraryResult
        """
        n = len(self.x)
        x_min, x_max = self.x.min(), self.x.max()

        # Binwidth: Freedman-Diaconis rule
        if self.binwidth is None:
            iqr = np.subtract(*np.percentile(self.x, [75, 25]))
            s = 2.0 * iqr * n ** (-1.0 / 3.0)
            if s <= 0:
                s = 3.5 * np.std(self.x) * n ** (-1.0 / 3.0)
        else:
            s = float(self.binwidth)

        # Bins aligned so that cutoff is always a bin boundary
        left_edges = np.arange(self.cutoff, x_min - s, -s)[::-1]
        right_edges = np.arange(self.cutoff + s, x_max + 2.0 * s, s)
        bin_edges = np.unique(np.concatenate([left_edges, right_edges]))

        bin_counts, _ = np.histogram(self.x, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_densities = bin_counts / (n * s)

        # Bandwidth for local linear smoothing.
        # Must cover enough bins (≥7) for a well-conditioned HC variance.
        if self.bandwidth is None:
            h = max(7.0 * s, 1.84 * np.std(self.x) * n ** (-0.2))
        else:
            h = float(self.bandwidth)

        left_mask = bin_centers < self.cutoff
        right_mask = bin_centers >= self.cutoff

        # Degenerate case: observations only on one side
        if left_mask.sum() < 2 or right_mask.sum() < 2:
            return McCraryResult(
                theta=0.0,
                se=np.inf,
                t_stat=0.0,
                p_value=1.0,
                bin_centers=bin_centers,
                bin_counts=bin_counts,
                fitted_left=bin_densities[left_mask],
                fitted_right=bin_densities[right_mask],
                conclusion="No evidence of manipulation",
            )

        f_left, var_fl, fitted_l = _wls_local_linear(
            bin_centers[left_mask], bin_densities[left_mask], self.cutoff, h
        )
        f_right, var_fr, fitted_r = _wls_local_linear(
            bin_centers[right_mask], bin_densities[right_mask], self.cutoff, h
        )

        # theta: log-difference in fitted densities at the cutoff (for display)
        f_left_safe = max(f_left, 1e-10)
        f_right_safe = max(f_right, 1e-10)
        theta = np.log(f_right_safe) - np.log(f_left_safe)

        # Test statistic: level difference f_right - f_left avoids the delta-method
        # blow-up when manipulation depletes f_left to near zero.
        diff = f_right - f_left
        se_diff = np.sqrt(var_fl + var_fr)
        se_diff = max(se_diff, 1e-12)

        t_stat = diff / se_diff
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        conclusion = (
            "Manipulation detected"
            if p_value < 0.05
            else "No evidence of manipulation"
        )

        return McCraryResult(
            theta=float(theta),
            se=float(se_diff),
            t_stat=float(t_stat),
            p_value=p_value,
            bin_centers=bin_centers,
            bin_counts=bin_counts,
            fitted_left=fitted_l,
            fitted_right=fitted_r,
            conclusion=conclusion,
        )
