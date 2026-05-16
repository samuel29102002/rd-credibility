"""Diagnostics module for analyzing R&D credibility."""

from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult, BandwidthSensitivity
from rd_credibility.diagnostics.covariate_balance import CovariateBalance, CovariateBalanceResult
from rd_credibility.diagnostics.donut import DonutRD, DonutResult
from rd_credibility.diagnostics.mccrary import McCraryResult, McCraryTest
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult, PlaceboTest

__all__ = [
    "McCraryTest",
    "McCraryResult",
    "CovariateBalance",
    "CovariateBalanceResult",
    "PlaceboTest",
    "PlaceboResult",
    "DonutRD",
    "DonutResult",
    "BandwidthSensitivity",
    "BandwidthGridResult",
]
