"""Estimation module for R&D credibility metrics."""

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth, rule_of_thumb_bandwidth
from rd_credibility.estimation.kernels import get_kernel
from rd_credibility.estimation.rdrobust import RDEstimator, RDResult

__all__ = [
    "RDEstimator",
    "RDResult",
    "get_kernel",
    "mse_optimal_bandwidth",
    "rule_of_thumb_bandwidth",
]
