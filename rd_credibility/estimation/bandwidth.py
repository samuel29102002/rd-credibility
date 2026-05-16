"""Bandwidth selection for local polynomial RD estimation."""

import numpy as np

from rd_credibility.estimation.kernels import get_kernel


def _local_poly_fit(y, x, weights, order):
    """
    Weighted least squares local polynomial fit.

    Returns
    -------
    coeffs : np.ndarray
        Polynomial coefficients (intercept first).
    """
    n = len(x)
    X = np.column_stack([x**p for p in range(order + 1)])
    W = np.diag(weights)
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y
    try:
        coeffs = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    return coeffs


def _estimate_density_at_cutoff(x, cutoff, bandwidth):
    """Estimate density of x at the cutoff using a histogram approach."""
    n = len(x)
    count = np.sum(np.abs(x - cutoff) <= bandwidth)
    return count / (2.0 * bandwidth * n)


def _estimate_conditional_variance(y, x, cutoff, bandwidth, side):
    """
    Estimate conditional variance at the cutoff from one side.

    Uses residuals from a local linear fit within the pilot bandwidth.
    """
    if side == "left":
        mask = (x >= cutoff - bandwidth) & (x < cutoff)
    else:
        mask = (x >= cutoff) & (x <= cutoff + bandwidth)

    xs = x[mask] - cutoff
    ys = y[mask]

    if len(xs) < 5:
        return np.var(y[np.abs(x - cutoff) <= bandwidth])

    X = np.column_stack([np.ones(len(xs)), xs])
    try:
        coeffs = np.linalg.lstsq(X, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.var(ys)

    residuals = ys - X @ coeffs
    return np.mean(residuals**2)


def _estimate_second_derivative(y, x, cutoff, pilot_bw, side):
    """
    Estimate the second derivative of the conditional mean at the cutoff.

    Uses a local cubic fit on the specified side within pilot_bw.
    """
    if side == "left":
        mask = (x >= cutoff - pilot_bw) & (x < cutoff)
    else:
        mask = (x >= cutoff) & (x <= cutoff + pilot_bw)

    xs = x[mask] - cutoff
    ys = y[mask]

    if len(xs) < 10:
        return 0.0

    # Local cubic: y = b0 + b1*x + b2*x^2 + b3*x^3
    X = np.column_stack([np.ones(len(xs)), xs, xs**2, xs**3])
    try:
        coeffs = np.linalg.lstsq(X, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0

    # Second derivative at cutoff (x=0): 2*b2
    return 2.0 * coeffs[2]


def _estimate_third_derivative(y, x, cutoff, pilot_bw, side):
    """
    Estimate the third derivative of the conditional mean at the cutoff.

    Uses a local quartic fit on the specified side within pilot_bw.
    """
    if side == "left":
        mask = (x >= cutoff - pilot_bw) & (x < cutoff)
    else:
        mask = (x >= cutoff) & (x <= cutoff + pilot_bw)

    xs = x[mask] - cutoff
    ys = y[mask]

    if len(xs) < 15:
        return 0.0

    X = np.column_stack([np.ones(len(xs)), xs, xs**2, xs**3, xs**4])
    try:
        coeffs = np.linalg.lstsq(X, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0

    # Third derivative at cutoff (x=0): 6*b3
    return 6.0 * coeffs[3]


def rule_of_thumb_bandwidth(y, x, cutoff=0.0):
    """
    Rule-of-thumb bandwidth selector for RD estimation.

    Uses a global polynomial fit to approximate curvature and Silverman's
    plug-in approach scaled for the RD context.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    x : array_like
        Running variable.
    cutoff : float, optional
        RD cutoff. Default is 0.0.

    Returns
    -------
    float
        The rule-of-thumb bandwidth.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Use IQR-based scale
    x_std = np.std(x)
    x_iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
    scale = min(x_std, x_iqr) if x_iqr > 0 else x_std

    # Silverman-type bandwidth scaled for RD (n^{-1/5} rate)
    h = 1.06 * scale * n ** (-1.0 / 5.0)

    # Ensure bandwidth doesn't exceed data range on either side
    x_range_left = cutoff - x.min()
    x_range_right = x.max() - cutoff
    h = min(h, 0.5 * x_range_left, 0.5 * x_range_right)

    return max(h, 1e-6)


def mse_optimal_bandwidth(y, x, cutoff=0.0, kernel="triangular"):
    """
    MSE-optimal bandwidth following Calonico, Cattaneo, and Titiunik (2014).

    Implements the CCT plug-in bandwidth selector for sharp RD designs with
    a local linear estimator. The procedure is:

    1. Compute a pilot bandwidth to estimate curvature (second derivatives).
    2. Estimate second derivatives of the conditional mean from each side.
    3. Estimate conditional variances at the cutoff from each side.
    4. Estimate density of the running variable at the cutoff.
    5. Plug into the MSE-optimal formula.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    x : array_like
        Running variable.
    cutoff : float, optional
        RD cutoff. Default is 0.0.
    kernel : str, optional
        Kernel name for the final estimator. Affects the kernel constants.
        Default is 'triangular'.

    Returns
    -------
    float
        The MSE-optimal bandwidth.

    References
    ----------
    Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust
    Nonparametric Confidence Intervals for Regression-Discontinuity Designs.
    Econometrica, 82(6), 2295-2326.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Kernel constants for local linear (p=1) estimation
    # C_K depends on the kernel: integral of K^2, integral of u^2*K(u), etc.
    kernel_constants = {
        "triangular": {"nu": 1 / 6, "mu2": 1 / 6, "kappa": 2 / 3},
        "uniform": {"nu": 1 / 2, "mu2": 1 / 3, "kappa": 1 / 2},
        "epanechnikov": {"nu": 3 / 5, "mu2": 1 / 5, "kappa": 3 / 5},
    }
    kc = kernel_constants.get(kernel, kernel_constants["triangular"])

    # Step 1: Pilot bandwidth for curvature estimation
    # Use a larger bandwidth (rate n^{-1/7} for estimating second derivative)
    h_pilot = rule_of_thumb_bandwidth(y, x, cutoff) * n ** (1.0 / 5.0 - 1.0 / 7.0)

    # Ensure pilot bandwidth has enough observations
    n_left_pilot = np.sum((x >= cutoff - h_pilot) & (x < cutoff))
    n_right_pilot = np.sum((x >= cutoff) & (x <= cutoff + h_pilot))

    if n_left_pilot < 10 or n_right_pilot < 10:
        h_pilot = rule_of_thumb_bandwidth(y, x, cutoff) * 2.0
        n_left_pilot = np.sum((x >= cutoff - h_pilot) & (x < cutoff))
        n_right_pilot = np.sum((x >= cutoff) & (x <= cutoff + h_pilot))
        if n_left_pilot < 10 or n_right_pilot < 10:
            return rule_of_thumb_bandwidth(y, x, cutoff)

    # Step 2: Estimate second derivatives (curvature) from each side
    m2_left = _estimate_second_derivative(y, x, cutoff, h_pilot, "left")
    m2_right = _estimate_second_derivative(y, x, cutoff, h_pilot, "right")

    # Step 3: Estimate conditional variances at the cutoff
    sigma2_left = _estimate_conditional_variance(y, x, cutoff, h_pilot, "left")
    sigma2_right = _estimate_conditional_variance(y, x, cutoff, h_pilot, "right")

    # Step 4: Density at the cutoff
    f_cutoff = _estimate_density_at_cutoff(x, cutoff, h_pilot)
    if f_cutoff < 1e-10:
        return rule_of_thumb_bandwidth(y, x, cutoff)

    # Step 5: MSE-optimal bandwidth formula
    # For local linear (p=1), the squared bias involves (m''_+ - m''_-) / 2
    # h_MSE = C * [ (sigma2_+ + sigma2_-) / (f * n * B^2) ]^{1/5}
    # where B = mu2 * (m''_+ - m''_-) / 2

    bias_term = kc["mu2"] * (m2_right - m2_left) / 2.0
    bias_sq = bias_term**2

    if bias_sq < 1e-10:
        # No detectable curvature difference — fall back
        return rule_of_thumb_bandwidth(y, x, cutoff)

    variance_term = kc["nu"] * (sigma2_left + sigma2_right) / f_cutoff

    # h_MSE = (variance_term / (2 * bias_sq * n))^{1/5}
    h_mse = (variance_term / (2.0 * bias_sq * n)) ** (1.0 / 5.0)

    # Bound the bandwidth to a reasonable range
    x_range = x.max() - x.min()
    h_mse = max(h_mse, x_range * 0.01)
    h_mse = min(h_mse, x_range * 0.5)

    return h_mse
