"""Synthetic regression discontinuity data generators for testing and benchmarking."""

import numpy as np
import pandas as pd


def generate_rd_data(
    n,
    cutoff=0.0,
    tau=1.0,
    slope_left=1.0,
    slope_right=1.0,
    noise=0.5,
    seed=None,
):
    """
    Generate synthetic sharp regression discontinuity data.

    The running variable is drawn from Uniform(-1, 1). Treatment is assigned
    sharply at the cutoff. Potential outcomes follow a piecewise-linear
    control function that is continuous at the cutoff, with a discontinuous
    jump of size ``tau`` under treatment.

    Parameters
    ----------
    n : int
        Number of observations.
    cutoff : float, optional
        Threshold value of the running variable for treatment assignment.
        Default is 0.0.
    tau : float, optional
        True treatment effect at the cutoff. Default is 1.0.
    slope_left : float, optional
        Slope of the control function for x < cutoff. Default is 1.0.
    slope_right : float, optional
        Slope of the control function for x >= cutoff. Default is 1.0.
    noise : float, optional
        Standard deviation of additive Gaussian noise. Default is 0.5.
    seed : int or None, optional
        Seed for the NumPy random generator. Default is None.

    Returns
    -------
    pd.DataFrame
        Columns: x, d, y, y0, y1.

        - **x** : running variable, Uniform(-1, 1)
        - **d** : treatment indicator, 1 if x >= cutoff else 0
        - **y** : observed outcome
        - **y0** : potential outcome under control (ground truth)
        - **y1** : potential outcome under treatment (ground truth)

    Notes
    -----
    The control function is continuous at the cutoff by construction:

        f(x) = slope_left * x                             if x < cutoff
        f(x) = (slope_left - slope_right)*cutoff
                + slope_right * x                          if x >= cutoff

    The RD estimand is E[Y1 - Y0 | X = cutoff] = tau.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1.0, 1.0, size=n)
    d = (x >= cutoff).astype(np.int8)
    epsilon = rng.normal(0.0, noise, size=n)

    # Piecewise-linear control function, continuous at cutoff
    alpha_right = (slope_left - slope_right) * cutoff
    f = np.where(x < cutoff, slope_left * x, alpha_right + slope_right * x)

    y0 = f + epsilon
    y1 = y0 + tau
    y = np.where(d == 1, y1, y0)

    return pd.DataFrame({"x": x, "d": d, "y": y, "y0": y0, "y1": y1})


def generate_rd_data_with_covariates(
    n,
    cutoff=0.0,
    tau=1.0,
    n_covariates=3,
    balance=True,
    seed=None,
):
    """
    Generate RD data augmented with pre-treatment covariates.

    Covariates are either balanced (no discontinuity at the cutoff, consistent
    with a valid design) or imbalanced (a mean shift at the cutoff, simulating
    a failed design where the assignment mechanism is compromised).

    Parameters
    ----------
    n : int
        Number of observations.
    cutoff : float, optional
        Threshold for treatment assignment. Default is 0.0.
    tau : float, optional
        True treatment effect at the cutoff. Default is 1.0.
    n_covariates : int, optional
        Number of pre-treatment covariates to generate. Default is 3.
    balance : bool, optional
        If True, covariates have identical distributions on both sides of
        the cutoff (valid design). If False, covariates have a mean shift
        at the cutoff (failed design). Default is True.
    seed : int or None, optional
        Seed for the NumPy random generator. Default is None.

    Returns
    -------
    pd.DataFrame
        All columns from :func:`generate_rd_data` plus z0, z1, ...,
        z{n_covariates-1} for the pre-treatment covariates.
    """
    rng = np.random.default_rng(seed)

    base = generate_rd_data(
        n, cutoff=cutoff, tau=tau, seed=int(rng.integers(0, 2**31))
    )
    d = base["d"].values

    covariate_frames = {}
    for k in range(n_covariates):
        if balance:
            z = rng.normal(0.0, 1.0, size=n)
        else:
            # Mean jumps by a random amount at the cutoff
            jump = rng.uniform(0.5, 1.5)
            z = rng.normal(0.0, 1.0, size=n) + jump * d
        covariate_frames[f"z{k}"] = z

    return pd.concat([base, pd.DataFrame(covariate_frames)], axis=1)


def generate_manipulation_data(n, cutoff=0.0, manipulation_frac=0.2, seed=None):
    """
    Generate RD data with strategic sorting above the cutoff.

    A fraction of observations that fall just below the cutoff are shifted
    to just above it, simulating agents who manipulate the running variable
    to receive treatment. This induces a discontinuity in the density of
    the running variable at the cutoff, violating the smoothness assumption
    required for RD identification.

    Parameters
    ----------
    n : int
        Number of observations.
    cutoff : float, optional
        Threshold for treatment assignment. Default is 0.0.
    manipulation_frac : float, optional
        Fraction of observations in the near-below window (within 0.2 units
        below the cutoff) that are shifted above the cutoff. Must be in
        [0, 1]. Default is 0.2.
    seed : int or None, optional
        Seed for the NumPy random generator. Default is None.

    Returns
    -------
    pd.DataFrame
        Columns: x, d, y, y0, y1, manipulated.

        - **x** : running variable after potential manipulation
        - **d** : treatment indicator based on post-manipulation x
        - **y** : observed outcome
        - **y0** : potential outcome under control
        - **y1** : potential outcome under treatment
        - **manipulated** : bool, True if the observation was shifted
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1.0, 1.0, size=n)

    bandwidth = 0.2
    near_below_mask = (x >= cutoff - bandwidth) & (x < cutoff)
    near_below_idx = np.where(near_below_mask)[0]

    n_to_shift = int(manipulation_frac * len(near_below_idx))
    shift_idx = rng.choice(near_below_idx, size=n_to_shift, replace=False)

    manipulated = np.zeros(n, dtype=bool)
    manipulated[shift_idx] = True

    # Place shifted units uniformly in a small window just above the cutoff
    x[shift_idx] = cutoff + rng.uniform(0.0, 0.05, size=n_to_shift)

    d = (x >= cutoff).astype(np.int8)
    epsilon = rng.normal(0.0, 0.5, size=n)

    y0 = x + epsilon
    y1 = y0 + 1.0
    y = np.where(d == 1, y1, y0)

    return pd.DataFrame(
        {"x": x, "d": d, "y": y, "y0": y0, "y1": y1, "manipulated": manipulated}
    )
