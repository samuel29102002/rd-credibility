"""Bandwidth and polynomial order sensitivity grid for RD estimation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator


@dataclass
class BandwidthGridResult:
    """
    Results from the bandwidth/polynomial sensitivity grid.

    Attributes
    ----------
    grid : pd.DataFrame
        One row per (bandwidth, poly_order) combination with columns:
        bandwidth, poly_order, estimate, se, ci_lower, ci_upper.
        Failed fits contain NaN.
    optimal_bandwidth : float
        MSE-optimal bandwidth from the CCT selector.
    cv_of_estimates : float
        Coefficient of variation of estimates across the 20 bandwidths
        with poly_order == 1 (measures sensitivity to bandwidth choice).
    stable_region : tuple of (float, float)
        Bandwidth range where local-linear estimates lie within
        1.96 * median(SE) of the median estimate.
    """

    grid: pd.DataFrame
    optimal_bandwidth: float
    cv_of_estimates: float
    stable_region: tuple


class BandwidthSensitivity:
    """
    Sensitivity analysis over a grid of bandwidths and polynomial orders.

    Estimates the RD effect for all (bandwidth, poly_order) combinations
    on a 20-point log-spaced bandwidth grid, producing a concise picture
    of how the estimate behaves as the bandwidth changes.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    x : array_like
        Running variable.
    cutoff : float, optional
        RD threshold. Default 0.
    bw_range : tuple of (float, float) or None, optional
        ``(bw_min, bw_max)`` for the bandwidth grid.  Defaults to
        ``(0.25 * h_CCT, 4 * h_CCT)`` clipped to the data range.
    poly_orders : list of int or None, optional
        Polynomial orders to include.  Default [1, 2, 3].
    """

    def __init__(self, y, x, cutoff=0, bw_range=None, poly_orders=None):
        self.y = np.asarray(y, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.bw_range = bw_range
        self.poly_orders = list(poly_orders) if poly_orders is not None else [1, 2, 3]

    def fit(self) -> BandwidthGridResult:
        """
        Run the bandwidth/polynomial sensitivity grid.

        Returns
        -------
        BandwidthGridResult
        """
        h_opt = mse_optimal_bandwidth(self.y, self.x, self.cutoff)
        x_range = float(self.x.max() - self.x.min())

        if self.bw_range is None:
            bw_min = max(h_opt * 0.25, x_range * 0.01)
            bw_max = min(h_opt * 4.0, x_range * 0.90)
        else:
            bw_min, bw_max = float(self.bw_range[0]), float(self.bw_range[1])

        bw_grid = np.logspace(np.log10(bw_min), np.log10(bw_max), 20)

        rows = []
        for bw in bw_grid:
            for p in self.poly_orders:
                try:
                    res = RDEstimator(
                        self.y,
                        self.x,
                        cutoff=self.cutoff,
                        bandwidth=float(bw),
                        poly_order=int(p),
                    ).fit()
                    rows.append(
                        {
                            "bandwidth": float(bw),
                            "poly_order": int(p),
                            "estimate": res.estimate,
                            "se": res.se,
                            "ci_lower": res.ci_lower,
                            "ci_upper": res.ci_upper,
                        }
                    )
                except Exception:
                    rows.append(
                        {
                            "bandwidth": float(bw),
                            "poly_order": int(p),
                            "estimate": np.nan,
                            "se": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                        }
                    )

        grid = pd.DataFrame(rows)

        # CV of estimates for poly_order == 1
        order1 = grid[grid["poly_order"] == 1].dropna(subset=["estimate"])
        if len(order1) > 1 and abs(order1["estimate"].mean()) > 1e-10:
            cv = float(order1["estimate"].std() / abs(order1["estimate"].mean()))
        else:
            cv = float("nan")

        # Stable region: bandwidths where local-linear estimate is within
        # 1.96 * median(SE) of the overall median estimate
        if len(order1) > 1:
            med_est = float(order1["estimate"].median())
            threshold = 1.96 * float(order1["se"].median())
            stable_bws = order1.loc[
                (order1["estimate"] - med_est).abs() < threshold, "bandwidth"
            ]
            if len(stable_bws) > 0:
                stable_region = (float(stable_bws.min()), float(stable_bws.max()))
            else:
                stable_region = (float(order1["bandwidth"].min()), float(order1["bandwidth"].max()))
        else:
            stable_region = (float(bw_min), float(bw_max))

        return BandwidthGridResult(
            grid=grid,
            optimal_bandwidth=float(h_opt),
            cv_of_estimates=cv,
            stable_region=stable_region,
        )
