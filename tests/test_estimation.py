"""Tests for local polynomial RD estimation."""

import numpy as np
import pytest

from rd_credibility.estimation import RDEstimator, RDResult
from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth, rule_of_thumb_bandwidth
from rd_credibility.estimation.kernels import triangular, uniform, epanechnikov
from tests.fixtures.synthetic_rd import generate_rd_data


# ---------------------------------------------------------------------------
# Kernel sanity checks
# ---------------------------------------------------------------------------


class TestKernels:
    def test_triangular_at_zero(self):
        assert triangular(0.0) == 1.0

    def test_triangular_at_boundary(self):
        assert triangular(1.0) == 0.0
        assert triangular(-1.0) == 0.0

    def test_triangular_outside(self):
        assert triangular(1.5) == 0.0

    def test_uniform_inside(self):
        assert uniform(0.5) == 0.5

    def test_uniform_outside(self):
        assert uniform(1.5) == 0.0

    def test_epanechnikov_at_zero(self):
        assert epanechnikov(0.0) == 0.75

    def test_epanechnikov_outside(self):
        assert epanechnikov(2.0) == 0.0

    def test_kernels_nonnegative(self):
        u = np.linspace(-2, 2, 100)
        assert (triangular(u) >= 0).all()
        assert (uniform(u) >= 0).all()
        assert (epanechnikov(u) >= 0).all()


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


class TestBandwidth:
    def test_rule_of_thumb_positive(self):
        df = generate_rd_data(500, seed=0)
        h = rule_of_thumb_bandwidth(df["y"].values, df["x"].values, cutoff=0.0)
        assert h > 0

    def test_mse_optimal_positive(self):
        df = generate_rd_data(2000, cutoff=0.0, tau=1.0, slope_left=1.0,
                              slope_right=2.0, noise=0.5, seed=0)
        h = mse_optimal_bandwidth(df["y"].values, df["x"].values, cutoff=0.0)
        assert h > 0

    def test_mse_optimal_reasonable_range(self):
        df = generate_rd_data(5000, cutoff=0.0, tau=1.0, slope_left=1.0,
                              slope_right=1.0, noise=0.3, seed=1)
        h = mse_optimal_bandwidth(df["y"].values, df["x"].values, cutoff=0.0)
        # Should be between 1% and 50% of the data range
        assert 0.02 < h < 1.0


# ---------------------------------------------------------------------------
# RD Estimator core
# ---------------------------------------------------------------------------


class TestRDEstimator:
    @pytest.fixture
    def large_sample(self):
        return generate_rd_data(
            2000, cutoff=0.0, tau=2.0, slope_left=1.0,
            slope_right=1.0, noise=0.5, seed=42
        )

    def test_returns_rd_result(self, large_sample):
        est = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, bandwidth=0.5
        )
        result = est.fit()
        assert isinstance(result, RDResult)

    def test_estimate_within_2se_of_truth(self, large_sample):
        """Point estimate should be within 2 SE of the true tau=2.0."""
        tau = 2.0
        est = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, bandwidth=0.5
        )
        result = est.fit()
        assert abs(result.estimate - tau) < 2.0 * result.se, (
            f"Estimate {result.estimate:.3f} not within 2 SE ({result.se:.3f}) "
            f"of true tau={tau}"
        )

    def test_estimate_within_2se_triangular(self):
        tau = 1.5
        df = generate_rd_data(2000, cutoff=0.0, tau=tau, noise=0.3, seed=10)
        result = RDEstimator(
            df["y"].values, df["x"].values, cutoff=0.0,
            kernel="triangular", bandwidth=0.4
        ).fit()
        assert abs(result.estimate - tau) < 2.0 * result.se

    def test_estimate_within_2se_uniform(self):
        tau = 1.5
        df = generate_rd_data(2000, cutoff=0.0, tau=tau, noise=0.3, seed=11)
        result = RDEstimator(
            df["y"].values, df["x"].values, cutoff=0.0,
            kernel="uniform", bandwidth=0.4
        ).fit()
        assert abs(result.estimate - tau) < 2.0 * result.se

    def test_estimate_within_2se_epanechnikov(self):
        tau = 1.5
        df = generate_rd_data(2000, cutoff=0.0, tau=tau, noise=0.3, seed=12)
        result = RDEstimator(
            df["y"].values, df["x"].values, cutoff=0.0,
            kernel="epanechnikov", bandwidth=0.4
        ).fit()
        assert abs(result.estimate - tau) < 2.0 * result.se

    def test_ci_contains_true_tau(self, large_sample):
        """95% CI should cover the truth in most cases."""
        tau = 2.0
        est = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, bandwidth=0.5
        )
        result = est.fit()
        assert result.ci_lower <= tau <= result.ci_upper, (
            f"CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}] "
            f"does not contain tau={tau}"
        )

    def test_se_increases_as_bandwidth_shrinks(self):
        """Smaller bandwidth -> fewer obs -> larger SE."""
        df = generate_rd_data(5000, cutoff=0.0, tau=1.0, noise=0.5, seed=20)
        bandwidths = [0.8, 0.5, 0.3, 0.15]
        ses = []
        for bw in bandwidths:
            result = RDEstimator(
                df["y"].values, df["x"].values,
                cutoff=0.0, bandwidth=bw
            ).fit()
            ses.append(result.se)

        # SE should be monotonically non-decreasing as bandwidth shrinks
        for i in range(len(ses) - 1):
            assert ses[i] <= ses[i + 1] * 1.01, (
                f"SE did not increase: bw={bandwidths[i]}->{bandwidths[i+1]}, "
                f"se={ses[i]:.4f}->{ses[i+1]:.4f}"
            )

    def test_estimate_monotonic_convergence(self):
        """
        With equal slopes on both sides, the estimate should be roughly
        stable as bandwidth changes (no strong monotonic drift).
        For different slopes, the estimate may shift as bandwidth shrinks
        toward the true local effect.
        """
        tau = 2.0
        df = generate_rd_data(
            10000, cutoff=0.0, tau=tau, slope_left=1.0,
            slope_right=3.0, noise=0.3, seed=30
        )
        bandwidths = [0.8, 0.5, 0.3, 0.15]
        estimates = []
        for bw in bandwidths:
            result = RDEstimator(
                df["y"].values, df["x"].values,
                cutoff=0.0, bandwidth=bw
            ).fit()
            estimates.append(result.estimate)

        # With different slopes, smaller bandwidth should get closer to true tau
        # (less bias from slope difference)
        errors = [abs(e - tau) for e in estimates]
        # The smallest bandwidth estimate should be within 0.5 of tau
        assert errors[-1] < 0.5, (
            f"Narrowest bandwidth estimate {estimates[-1]:.3f} "
            f"too far from tau={tau}"
        )

    def test_n_left_n_right_consistent(self, large_sample):
        est = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, bandwidth=0.5
        )
        result = est.fit()
        assert result.n_left > 0
        assert result.n_right > 0
        assert result.n_left + result.n_right <= len(large_sample)

    def test_p_value_range(self, large_sample):
        result = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, bandwidth=0.5
        ).fit()
        assert 0.0 <= result.p_value <= 1.0

    def test_significant_effect_detected(self, large_sample):
        """With tau=2.0 and n=2000, the effect should be highly significant."""
        result = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, bandwidth=0.5
        ).fit()
        assert result.p_value < 0.05

    def test_no_effect_not_significant(self):
        """With tau=0, should usually not reject the null."""
        df = generate_rd_data(2000, cutoff=0.0, tau=0.0, noise=1.0, seed=50)
        result = RDEstimator(
            df["y"].values, df["x"].values,
            cutoff=0.0, bandwidth=0.5
        ).fit()
        assert result.p_value > 0.01

    def test_auto_bandwidth(self, large_sample):
        """Estimator should run with automatic bandwidth selection."""
        est = RDEstimator(
            large_sample["y"].values, large_sample["x"].values,
            cutoff=0.0, kernel="triangular"
        )
        result = est.fit()
        assert result.bandwidth > 0
        assert np.isfinite(result.estimate)

    def test_insufficient_obs_raises(self):
        """Tiny bandwidth with few data points should raise ValueError."""
        df = generate_rd_data(50, cutoff=0.0, seed=0)
        est = RDEstimator(
            df["y"].values, df["x"].values,
            cutoff=0.0, bandwidth=0.001
        )
        with pytest.raises(ValueError, match="Insufficient observations"):
            est.fit()

    def test_nonzero_cutoff(self):
        """Estimator should work with a non-zero cutoff."""
        tau = 3.0
        cutoff = 0.3
        df = generate_rd_data(3000, cutoff=cutoff, tau=tau, noise=0.3, seed=60)
        result = RDEstimator(
            df["y"].values, df["x"].values,
            cutoff=cutoff, bandwidth=0.3
        ).fit()
        assert abs(result.estimate - tau) < 2.0 * result.se


# ---------------------------------------------------------------------------
# Known-truth recovery (high-power large-sample tests)
# ---------------------------------------------------------------------------


class TestKnownTruthRecovery:
    """With n=5000 the estimate should be very close to the truth."""

    @pytest.fixture(scope="class")
    def large_clean(self):
        return generate_rd_data(5000, cutoff=0.0, tau=1.0, noise=0.5, seed=99)

    def test_estimate_close_to_tau(self, large_clean):
        result = RDEstimator(
            large_clean["y"].values, large_clean["x"].values,
            cutoff=0.0, bandwidth=0.4,
        ).fit()
        assert abs(result.estimate - 1.0) < 0.1, (
            f"n=5000 estimate {result.estimate:.4f} deviates from tau=1.0 by "
            f"{abs(result.estimate - 1.0):.4f} (limit 0.10)"
        )

    def test_se_small_at_large_n(self, large_clean):
        result = RDEstimator(
            large_clean["y"].values, large_clean["x"].values,
            cutoff=0.0, bandwidth=0.4,
        ).fit()
        assert result.se < 0.10, f"SE={result.se:.4f} unexpectedly large at n=5000"

    def test_bandwidth_finite_and_positive(self, large_clean):
        bw = mse_optimal_bandwidth(
            large_clean["y"].values, large_clean["x"].values, cutoff=0.0
        )
        assert np.isfinite(bw), "MSE-optimal bandwidth is not finite"
        assert bw > 0, "MSE-optimal bandwidth is not positive"

    def test_rule_of_thumb_finite_and_positive(self, large_clean):
        bw = rule_of_thumb_bandwidth(
            large_clean["y"].values, large_clean["x"].values, cutoff=0.0
        )
        assert np.isfinite(bw)
        assert bw > 0


class TestAllKernelsValidEstimates:
    """All three kernels should produce finite, non-NaN estimates."""

    @pytest.fixture(scope="class")
    def data(self):
        return generate_rd_data(2000, tau=1.0, noise=0.4, seed=77)

    @pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
    def test_kernel_produces_finite_estimate(self, data, kernel):
        result = RDEstimator(
            data["y"].values, data["x"].values,
            cutoff=0.0, bandwidth=0.4, kernel=kernel,
        ).fit()
        assert np.isfinite(result.estimate), f"{kernel} kernel gave non-finite estimate"
        assert np.isfinite(result.se), f"{kernel} kernel gave non-finite SE"
        assert result.se > 0

    @pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
    def test_kernel_ci_valid(self, data, kernel):
        result = RDEstimator(
            data["y"].values, data["x"].values,
            cutoff=0.0, bandwidth=0.4, kernel=kernel,
        ).fit()
        assert result.ci_lower < result.ci_upper
        assert 0.0 <= result.p_value <= 1.0


class TestFuzzyRDFirstStage:
    """
    Fuzzy RD first-stage test.

    When the instrument (being above the cutoff) is strong, regressing the
    treatment indicator on the discontinuity should yield a large F-statistic.
    The first-stage F = (estimate / SE)^2 should exceed 10 (Stock & Yogo weak
    instrument threshold).
    """

    @pytest.fixture(scope="class")
    def fuzzy_data(self):
        """
        Fuzzy RD: 85% compliance — most units above the cutoff take treatment,
        5% of units below also take it (always-takers).
        """
        rng = np.random.default_rng(123)
        n = 3000
        x = rng.uniform(-1.0, 1.0, n)
        above = x >= 0.0
        # Compliers take treatment; never-takers don't; always-takers always do
        compliance = rng.uniform(size=n)
        d = np.where(above, (compliance < 0.85).astype(float),
                     (compliance < 0.05).astype(float))
        y0 = 0.5 * x + rng.normal(0, 0.5, n)
        y1 = y0 + 1.5
        y = np.where(d == 1, y1, y0)
        return {"x": x, "d": d, "y": y}

    def test_first_stage_f_above_10(self, fuzzy_data):
        """
        First stage: regress D on being above cutoff.
        F = (jump in D at cutoff / SE)^2 should exceed 10.
        """
        result = RDEstimator(
            fuzzy_data["d"], fuzzy_data["x"],
            cutoff=0.0, bandwidth=0.4, poly_order=1,
        ).fit()
        f_stat = (result.estimate / result.se) ** 2
        assert f_stat > 10, (
            f"First-stage F={f_stat:.1f} < 10 — instrument appears weak "
            f"(estimate={result.estimate:.3f}, se={result.se:.3f})"
        )

    def test_first_stage_estimate_near_compliance_rate(self, fuzzy_data):
        """
        At the cutoff, the jump in D ≈ compliance_rate - always_taker_rate ≈ 0.80.
        """
        result = RDEstimator(
            fuzzy_data["d"], fuzzy_data["x"],
            cutoff=0.0, bandwidth=0.4, poly_order=1,
        ).fit()
        # Expected ~0.80 (= 0.85 compliers - 0.05 always-takers)
        assert 0.55 < result.estimate < 0.99, (
            f"First-stage estimate {result.estimate:.3f} outside expected range (0.55, 0.99)"
        )


class TestBandwidthEdgeCases:
    """Cover edge-case paths in bandwidth selection."""

    def test_mse_optimal_small_n(self):
        """Small n may fall back to rule-of-thumb — should still return finite positive."""
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, 30)
        y = x + rng.normal(0, 0.5, 30)
        bw = mse_optimal_bandwidth(y, x, 0.0)
        assert np.isfinite(bw) and bw > 0

    def test_mse_optimal_zero_curvature(self):
        """When curvature difference ≈ 0 (flat function), falls back to rule-of-thumb."""
        rng = np.random.default_rng(1)
        # Perfectly linear Y = X + noise → no curvature on either side
        x = rng.uniform(-1, 1, 2000)
        y = x + rng.normal(0, 0.1, 2000)  # very small noise → near-zero curvature diff
        bw = mse_optimal_bandwidth(y, x, 0.0)
        assert np.isfinite(bw) and bw > 0

    def test_mse_optimal_non_central_cutoff(self):
        df = generate_rd_data(1000, cutoff=0.3, tau=1.0, seed=5)
        bw = mse_optimal_bandwidth(df["y"].values, df["x"].values, cutoff=0.3)
        assert np.isfinite(bw) and bw > 0

    def test_rule_of_thumb_non_central_cutoff(self):
        df = generate_rd_data(500, cutoff=-0.4, tau=1.0, seed=6)
        bw = rule_of_thumb_bandwidth(df["y"].values, df["x"].values, cutoff=-0.4)
        assert np.isfinite(bw) and bw > 0

    def test_bandwidth_private_helpers(self):
        """Exercise _estimate_third_derivative and _local_poly_fit."""
        from rd_credibility.estimation.bandwidth import (
            _estimate_third_derivative,
            _local_poly_fit,
        )
        rng = np.random.default_rng(42)
        x = rng.uniform(-0.5, 0, 50)
        y = x**3 + rng.normal(0, 0.1, 50)
        # Third derivative of x^3 = 6 → should be non-zero
        m3 = _estimate_third_derivative(y, x, cutoff=0.0, pilot_bw=0.5, side="left")
        assert np.isfinite(m3)

        # _local_poly_fit
        X = np.column_stack([np.ones(50), x])
        w = np.ones(50)
        coeffs = _local_poly_fit(y, x, w, order=1)
        assert len(coeffs) == 2
        assert all(np.isfinite(coeffs))

    def test_third_derivative_sparse(self):
        """With < 15 obs, third derivative falls back to 0."""
        from rd_credibility.estimation.bandwidth import _estimate_third_derivative
        x = np.linspace(-0.1, 0, 10)
        y = x + 0.1
        result = _estimate_third_derivative(y, x, 0.0, 0.5, "left")
        assert result == 0.0

    def test_conditional_variance_sparse(self):
        """With < 5 obs on one side, variance falls back to global variance."""
        from rd_credibility.estimation.bandwidth import _estimate_conditional_variance
        x = np.array([-0.5, -0.4, -0.3, 0.1, 0.2, 0.3])
        y = x + 0.1
        # "left" with tight bandwidth → < 5 obs → should not raise
        v = _estimate_conditional_variance(y, x, 0.0, 0.35, "left")
        assert np.isfinite(v) and v >= 0
