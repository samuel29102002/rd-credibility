"""Local polynomial RD estimation."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.kernels import get_kernel


@dataclass
class RDResult:
    """
    Results from a local polynomial RD estimation.

    Attributes
    ----------
    estimate : float
        The RD point estimate (treatment effect at the cutoff).
    se : float
        Heteroskedasticity-robust (HC1) standard error.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    bandwidth : float
        Bandwidth used for estimation.
    n_left : int
        Number of observations within bandwidth to the left of the cutoff.
    n_right : int
        Number of observations within bandwidth to the right of the cutoff.
    p_value : float
        Two-sided p-value for the null hypothesis of zero treatment effect.
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    bandwidth: float
    n_left: int
    n_right: int
    p_value: float


def _weighted_least_squares(X, y, w):
    """
    Solve weighted least squares: min_b sum w_i (y_i - X_i b)^2.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    w : np.ndarray, shape (n,)
        Non-negative weights.

    Returns
    -------
    coeffs : np.ndarray, shape (p,)
        Estimated coefficients.
    XtWX_inv : np.ndarray, shape (p, p)
        Inverse of X'WX (needed for variance computation).
    residuals : np.ndarray, shape (n,)
        Residuals y - X @ coeffs.
    """
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    XtWX = Xw.T @ Xw
    XtWy = Xw.T @ yw

    try:
        XtWX_inv = np.linalg.inv(XtWX)
        coeffs = XtWX_inv @ XtWy
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
        XtWX_inv = np.linalg.pinv(XtWX)

    residuals = y - X @ coeffs
    return coeffs, XtWX_inv, residuals


def _hc_robust_variance(X, w, residuals, XtWX_inv):
    """
    HC1 heteroskedasticity-robust variance estimator for WLS.

    Computes: (X'WX)^{-1} X'W diag(e^2) W X (X'WX)^{-1} * n/(n-p)

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    w : np.ndarray, shape (n,)
    residuals : np.ndarray, shape (n,)
    XtWX_inv : np.ndarray, shape (p, p)

    Returns
    -------
    np.ndarray, shape (p, p)
        HC1 variance-covariance matrix of coefficients.
    """
    n, p = X.shape
    Xw = X * w[:, None]  # X * W (each row scaled by w_i)
    meat = Xw.T @ np.diag(residuals**2) @ Xw
    V = XtWX_inv @ meat @ XtWX_inv
    # HC1 small-sample correction
    V *= n / max(n - p, 1)
    return V


class RDEstimator:
    """
    Local polynomial regression discontinuity estimator.

    Estimates the average treatment effect at the cutoff using weighted
    local polynomial regression from each side, with heteroskedasticity-
    robust standard errors.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    x : array_like
        Running variable.
    cutoff : float, optional
        The RD cutoff threshold. Default is 0.
    kernel : str, optional
        Kernel for weighting. One of 'triangular', 'uniform', 'epanechnikov'.
        Default is 'triangular'.
    poly_order : int, optional
        Order of the local polynomial. Default is 1 (local linear).
    bandwidth : float or None, optional
        Bandwidth for estimation. If None, the MSE-optimal bandwidth is
        computed automatically. Default is None.
    """

    def __init__(self, y, x, cutoff=0, kernel="triangular", poly_order=1, bandwidth=None):
        self.y = np.asarray(y, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.kernel_name = kernel
        self.kernel_fn = get_kernel(kernel)
        self.poly_order = int(poly_order)
        self.bandwidth = bandwidth

    def fit(self):
        """
        Fit the local polynomial RD model.

        Returns
        -------
        RDResult
            Estimation results including point estimate, standard error,
            confidence interval, bandwidth, effective sample sizes, and p-value.

        Raises
        ------
        ValueError
            If there are insufficient observations within the bandwidth
            on either side of the cutoff.
        """
        # Bandwidth selection
        if self.bandwidth is None:
            h = mse_optimal_bandwidth(
                self.y, self.x, cutoff=self.cutoff, kernel=self.kernel_name
            )
        else:
            h = float(self.bandwidth)

        # Select observations within bandwidth
        x_centered = self.x - self.cutoff
        left_mask = (x_centered >= -h) & (x_centered < 0)
        right_mask = (x_centered >= 0) & (x_centered <= h)

        n_left = int(np.sum(left_mask))
        n_right = int(np.sum(right_mask))

        if n_left < self.poly_order + 1:
            raise ValueError(
                f"Insufficient observations left of cutoff: {n_left} "
                f"(need at least {self.poly_order + 1})"
            )
        if n_right < self.poly_order + 1:
            raise ValueError(
                f"Insufficient observations right of cutoff: {n_right} "
                f"(need at least {self.poly_order + 1})"
            )

        # Fit on each side separately
        mu_left, var_left = self._fit_side(x_centered[left_mask], self.y[left_mask], h)
        mu_right, var_right = self._fit_side(x_centered[right_mask], self.y[right_mask], h)

        # RD estimate: difference in predicted values at cutoff (x_centered=0)
        estimate = mu_right - mu_left

        # Combined SE (independent samples on each side)
        se = np.sqrt(var_left + var_right)

        # 95% CI and p-value
        z_crit = stats.norm.ppf(0.975)
        ci_lower = estimate - z_crit * se
        ci_upper = estimate + z_crit * se

        z_stat = estimate / se if se > 0 else np.inf
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))

        return RDResult(
            estimate=float(estimate),
            se=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            bandwidth=float(h),
            n_left=n_left,
            n_right=n_right,
            p_value=float(p_value),
        )

    def _fit_side(self, x_centered, y, h):
        """
        Fit local polynomial on one side and return predicted value and
        variance of the predicted value at x=0 (the cutoff).

        Parameters
        ----------
        x_centered : np.ndarray
            Running variable centered at the cutoff, for one side only.
        y : np.ndarray
            Outcomes for one side.
        h : float
            Bandwidth.

        Returns
        -------
        mu_hat : float
            Predicted value at x_centered=0.
        var_mu : float
            HC-robust variance of the predicted value at x_centered=0.
        """
        n = len(x_centered)
        p = self.poly_order

        # Kernel weights
        u = x_centered / h
        w = self.kernel_fn(u)

        # Design matrix: [1, x, x^2, ..., x^p]
        X = np.column_stack([x_centered**j for j in range(p + 1)])

        # WLS fit
        coeffs, XtWX_inv, residuals = _weighted_least_squares(X, y, w)

        # Predicted value at x_centered=0 is just the intercept
        mu_hat = coeffs[0]

        # HC-robust variance of the intercept
        V = _hc_robust_variance(X, w, residuals, XtWX_inv)
        var_mu = V[0, 0]

        return mu_hat, var_mu
