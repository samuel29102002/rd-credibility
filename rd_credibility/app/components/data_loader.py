"""Built-in dataset loaders and CSV upload handler."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lee_2008(seed=0) -> pd.DataFrame:
    """
    Lee (2008) electoral RD: vote share as running variable, re-election as outcome.
    Synthetic approximation preserving the known tau ≈ 0.40.
    """
    rng = np.random.default_rng(seed)
    n = 6558
    x = rng.uniform(-1, 1, n)
    x = np.sign(x) * np.abs(x) ** 0.7  # skew toward zero like real electoral margins
    d = (x >= 0).astype(float)

    # Smooth control function continuous at cutoff
    f = 0.48 + 0.42 * x - 0.05 * x ** 2
    y = f + 0.40 * d + rng.normal(0, 0.18, n)
    y = np.clip(y, 0, 1)

    # Pre-treatment covariates: lagged vote share, district characteristics
    z_lag = 0.50 + 0.35 * x + rng.normal(0, 0.12, n)
    z_turnout = 0.60 + 0.04 * x + rng.normal(0, 0.08, n)
    z_urban = rng.binomial(1, 0.55 + 0.02 * x, n).astype(float)

    return pd.DataFrame({
        "x": x, "d": d, "y": y,
        "z_lag_vote": z_lag, "z_turnout": z_turnout, "z_urban": z_urban,
    })


def _maimonides(seed=1) -> pd.DataFrame:
    """
    Angrist & Lavy (1999) Maimonides Rule: class size discontinuity at enrolment multiples of 40.
    Running variable is enrolment mod 40, cutoff at 0 (centred).
    """
    rng = np.random.default_rng(seed)
    n = 2019
    x = rng.uniform(-19, 20, n)
    d = (x >= 0).astype(float)

    # Class size jumps down (negative effect on class size = positive effect on scores)
    class_size = 28 - 8 * d + 0.12 * x + rng.normal(0, 2, n)
    y = 75 - 0.35 * class_size + rng.normal(0, 5, n)

    z_ses = rng.normal(50 + 0.1 * x, 10, n)        # socioeconomic status index
    z_pct_dis = rng.uniform(0, 0.5, n)               # pct disadvantaged

    return pd.DataFrame({
        "x": x, "d": d, "y": y,
        "z_ses": z_ses, "z_pct_dis": z_pct_dis,
    })


def _thistlethwaite(seed=2) -> pd.DataFrame:
    """
    Thistlethwaite & Campbell (1960) scholarship threshold.
    Running variable: test score centred at cutoff (0).
    Outcome: college attendance indicator.
    """
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.normal(0, 0.4, n)
    x = np.clip(x, -1, 1)
    d = (x >= 0).astype(float)

    p_attend = 0.32 + 0.38 * d + 0.25 * x - 0.08 * x ** 2
    p_attend = np.clip(p_attend, 0, 1)
    y = rng.binomial(1, p_attend).astype(float)

    z_ability = rng.normal(100 + 15 * x, 12, n)
    z_income = rng.lognormal(10 + 0.2 * x, 0.6, n)

    return pd.DataFrame({
        "x": x, "d": d, "y": y,
        "z_ability": z_ability, "z_income": z_income,
    })


_DATASETS = {
    "Lee 2008 (Electoral)": _lee_2008,
    "Maimonides Rule": _maimonides,
    "Thistlethwaite 1960": _thistlethwaite,
}

DEFAULT_CUTOFFS = {
    "Lee 2008 (Electoral)": 0.0,
    "Maimonides Rule": 0.0,
    "Thistlethwaite 1960": 0.0,
}


def load_builtin(name: str) -> pd.DataFrame:
    return _DATASETS[name]()


def covariate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("z")]
