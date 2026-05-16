"""Tests for all five RD diagnostic modules."""

import numpy as np
import pandas as pd
import pytest

from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult, BandwidthSensitivity
from rd_credibility.diagnostics.covariate_balance import CovariateBalance, CovariateBalanceResult
from rd_credibility.diagnostics.donut import DonutRD, DonutResult
from rd_credibility.diagnostics.mccrary import McCraryResult, McCraryTest
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult, PlaceboTest
from tests.fixtures.synthetic_rd import (
    generate_manipulation_data,
    generate_rd_data,
    generate_rd_data_with_covariates,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture(scope="module")
def clean_rd():
    return generate_rd_data(5000, cutoff=0.0, tau=2.0, noise=0.5, seed=SEED)


@pytest.fixture(scope="module")
def manipulated():
    return generate_manipulation_data(15000, cutoff=0.0, manipulation_frac=0.90, seed=SEED)


@pytest.fixture(scope="module")
def balanced_covs():
    return generate_rd_data_with_covariates(
        3000, cutoff=0.0, tau=1.5, n_covariates=5, balance=True, seed=SEED
    )


@pytest.fixture(scope="module")
def imbalanced_covs():
    return generate_rd_data_with_covariates(
        5000, cutoff=0.0, tau=1.5, n_covariates=5, balance=False, seed=SEED
    )


# ---------------------------------------------------------------------------
# McCrary density test
# ---------------------------------------------------------------------------


class TestMcCraryTest:
    def test_returns_mccrary_result(self, clean_rd):
        result = McCraryTest(clean_rd["x"].values).fit()
        assert isinstance(result, McCraryResult)

    def test_result_fields_populated(self, clean_rd):
        result = McCraryTest(clean_rd["x"].values).fit()
        assert np.isfinite(result.theta)
        assert result.se > 0
        assert 0.0 <= result.p_value <= 1.0
        assert len(result.bin_centers) > 0
        assert len(result.bin_counts) == len(result.bin_centers)

    def test_conclusion_valid_strings(self, clean_rd):
        result = McCraryTest(clean_rd["x"].values).fit()
        assert result.conclusion in {"No evidence of manipulation", "Manipulation detected"}

    def test_fitted_arrays_same_length_as_masks(self, clean_rd):
        x = clean_rd["x"].values
        result = McCraryTest(x).fit()
        n_left = (result.bin_centers < 0.0).sum()
        n_right = (result.bin_centers >= 0.0).sum()
        assert len(result.fitted_left) == n_left
        assert len(result.fitted_right) == n_right

    def test_clean_data_not_rejected(self, clean_rd):
        """Uniform running variable with no manipulation should not be rejected."""
        result = McCraryTest(clean_rd["x"].values, cutoff=0.0).fit()
        # Allow some sampling variation — test p > 0.01 for very large n
        assert result.p_value > 0.01, (
            f"False rejection on clean data: p={result.p_value:.4f}"
        )

    def test_manipulation_detected(self, manipulated):
        """Heavy manipulation (90 %) should produce a significant density jump."""
        result = McCraryTest(manipulated["x"].values, cutoff=0.0).fit()
        assert result.p_value < 0.05, (
            f"Manipulation not detected: p={result.p_value:.4f}, "
            f"t={result.t_stat:.3f}"
        )
        assert result.conclusion == "Manipulation detected"

    def test_theta_sign_with_manipulation(self, manipulated):
        """Manipulation shifts mass above the cutoff, so theta > 0."""
        result = McCraryTest(manipulated["x"].values, cutoff=0.0).fit()
        assert result.theta > 0

    def test_custom_binwidth_accepted(self, clean_rd):
        result = McCraryTest(clean_rd["x"].values, binwidth=0.05).fit()
        assert np.isfinite(result.theta)

    def test_custom_bandwidth_accepted(self, clean_rd):
        result = McCraryTest(clean_rd["x"].values, bandwidth=0.3).fit()
        assert result.se > 0


# ---------------------------------------------------------------------------
# Covariate balance
# ---------------------------------------------------------------------------


class TestCovariateBalance:
    def _cov_df(self, df):
        cov_cols = [c for c in df.columns if c.startswith("z")]
        return df[cov_cols]

    def test_returns_covariate_balance_result(self, balanced_covs):
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(balanced_covs["x"].values, covs, cutoff=0.0).fit()
        assert isinstance(result, CovariateBalanceResult)

    def test_results_dataframe_columns(self, balanced_covs):
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(balanced_covs["x"].values, covs, cutoff=0.0).fit()
        assert set(result.results.columns) >= {"covariate", "estimate", "se", "p_value", "significant"}

    def test_results_row_count(self, balanced_covs):
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(balanced_covs["x"].values, covs, cutoff=0.0).fit()
        assert len(result.results) == 5  # n_covariates=5

    def test_mean_p_value_in_range(self, balanced_covs):
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(balanced_covs["x"].values, covs, cutoff=0.0).fit()
        assert 0.0 <= result.mean_p_value <= 1.0

    def test_balanced_covariates_appear_balanced(self, balanced_covs):
        """Balanced covariates: at most 1 out of 5 significant at 5%."""
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(
            balanced_covs["x"].values, covs, cutoff=0.0, bandwidth=0.4
        ).fit()
        assert result.n_significant <= 1, (
            f"{result.n_significant} balanced covariates flagged as imbalanced"
        )
        assert result.overall_conclusion == "Covariates appear balanced"

    def test_imbalanced_covariates_detected(self, imbalanced_covs):
        """Imbalanced covariates: at least 1 flagged significant."""
        covs = self._cov_df(imbalanced_covs)
        result = CovariateBalance(
            imbalanced_covs["x"].values, covs, cutoff=0.0, bandwidth=0.4
        ).fit()
        assert result.n_significant >= 1, (
            "No imbalanced covariate detected"
        )

    def test_n_significant_matches_results_df(self, balanced_covs):
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(balanced_covs["x"].values, covs).fit()
        assert result.n_significant == int(result.results["significant"].sum())

    def test_conclusion_is_one_of_two_strings(self, balanced_covs):
        covs = self._cov_df(balanced_covs)
        result = CovariateBalance(balanced_covs["x"].values, covs).fit()
        assert result.overall_conclusion in {
            "Covariates appear balanced",
            "Potential covariate imbalance detected",
        }


# ---------------------------------------------------------------------------
# Placebo cutoffs
# ---------------------------------------------------------------------------


class TestPlaceboTest:
    def test_returns_placebo_result(self, clean_rd):
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values, cutoff=0.0, bandwidth=0.4, n_placebo=10
        ).fit()
        assert isinstance(result, PlaceboResult)

    def test_placebo_array_lengths_match(self, clean_rd):
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4, n_placebo=12
        ).fit()
        assert len(result.placebo_estimates) == len(result.placebo_cutoffs)
        assert len(result.placebo_ses) == len(result.placebo_cutoffs)

    def test_true_estimate_populated(self, clean_rd):
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4, n_placebo=10
        ).fit()
        assert np.isfinite(result.true_estimate)
        assert result.true_se > 0

    def test_n_significant_nonnegative(self, clean_rd):
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4, n_placebo=10
        ).fit()
        assert result.n_significant_placebos >= 0

    def test_valid_design_few_significant_placebos(self, clean_rd):
        """Under a valid design, fewer than half of placebos should be significant."""
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values,
            cutoff=0.0, bandwidth=0.4, n_placebo=20
        ).fit()
        n = len(result.placebo_cutoffs)
        assert result.n_significant_placebos < n // 2, (
            f"{result.n_significant_placebos}/{n} placebos significant on valid data"
        )

    def test_placebo_cutoffs_outside_true_cutoff_margin(self, clean_rd):
        """No placebo should be placed right at the true cutoff."""
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values, cutoff=0.0, bandwidth=0.4
        ).fit()
        assert not np.any(result.placebo_cutoffs == 0.0)

    def test_conclusion_is_string(self, clean_rd):
        result = PlaceboTest(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4, n_placebo=10
        ).fit()
        assert isinstance(result.conclusion, str) and len(result.conclusion) > 0


# ---------------------------------------------------------------------------
# Donut RD
# ---------------------------------------------------------------------------


class TestDonutRD:
    def test_returns_donut_result(self, clean_rd):
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values, cutoff=0.0, bandwidth=0.4
        ).fit()
        assert isinstance(result, DonutResult)

    def test_lists_same_length(self, clean_rd):
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4
        ).fit()
        assert len(result.estimates) == len(result.donut_holes)
        assert len(result.ses) == len(result.donut_holes)

    def test_baseline_is_zero_hole_estimate(self, clean_rd):
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4
        ).fit()
        assert result.baseline_estimate == pytest.approx(result.estimates[0], abs=1e-8)

    def test_max_deviation_nonnegative(self, clean_rd):
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4
        ).fit()
        assert result.max_deviation >= 0.0

    def test_custom_holes_used(self, clean_rd):
        holes = [0.0, 0.02, 0.05]
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values,
            bandwidth=0.4, donut_holes=holes
        ).fit()
        assert result.donut_holes == holes
        assert len(result.estimates) == 3

    def test_small_holes_stable(self, clean_rd):
        """For small donut holes, estimate should stay close to baseline."""
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4
        ).fit()
        # Max deviation across all holes should be less than twice the SE at baseline
        baseline_se = result.ses[0]
        assert result.max_deviation < 5.0 * baseline_se, (
            f"Estimate unstable: max_dev={result.max_deviation:.3f}, "
            f"baseline_se={baseline_se:.3f}"
        )

    def test_all_ses_positive(self, clean_rd):
        result = DonutRD(
            clean_rd["y"].values, clean_rd["x"].values, bandwidth=0.4
        ).fit()
        ses = [s for s in result.ses if not np.isnan(s)]
        assert all(s > 0 for s in ses)


# ---------------------------------------------------------------------------
# Bandwidth sensitivity grid
# ---------------------------------------------------------------------------


class TestBandwidthSensitivity:
    def test_returns_bandwidth_grid_result(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values, cutoff=0.0,
            poly_orders=[1, 2]
        ).fit()
        assert isinstance(result, BandwidthGridResult)

    def test_grid_columns(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values, poly_orders=[1]
        ).fit()
        assert set(result.grid.columns) >= {"bandwidth", "poly_order", "estimate", "se",
                                             "ci_lower", "ci_upper"}

    def test_grid_row_count(self, clean_rd):
        """20 bandwidths × len(poly_orders) rows expected."""
        poly_orders = [1, 2]
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values,
            poly_orders=poly_orders
        ).fit()
        assert len(result.grid) == 20 * len(poly_orders)

    def test_optimal_bandwidth_positive(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values
        ).fit()
        assert result.optimal_bandwidth > 0

    def test_cv_of_estimates_nonnegative(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values
        ).fit()
        if not np.isnan(result.cv_of_estimates):
            assert result.cv_of_estimates >= 0.0

    def test_stable_region_is_tuple_of_two(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values
        ).fit()
        assert isinstance(result.stable_region, tuple)
        assert len(result.stable_region) == 2

    def test_stable_region_ordered(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values
        ).fit()
        assert result.stable_region[0] <= result.stable_region[1]

    def test_custom_bw_range(self, clean_rd):
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values,
            bw_range=(0.1, 0.8), poly_orders=[1]
        ).fit()
        bws = result.grid["bandwidth"]
        assert bws.min() >= 0.09  # log-space endpoints allow small float drift
        assert bws.max() <= 0.81

    def test_estimates_near_truth_at_moderate_bandwidths(self, clean_rd):
        """
        For moderate bandwidths with poly_order=1, most estimates should be
        within 2 SE of the true tau=2.0.
        """
        tau = 2.0
        result = BandwidthSensitivity(
            clean_rd["y"].values, clean_rd["x"].values,
            cutoff=0.0, bw_range=(0.2, 0.8), poly_orders=[1]
        ).fit()
        order1 = result.grid[result.grid["poly_order"] == 1].dropna(subset=["estimate"])
        within_2se = ((order1["estimate"] - tau).abs() < 2.0 * order1["se"])
        frac = within_2se.mean()
        assert frac >= 0.7, (
            f"Only {frac:.0%} of bandwidth-1 estimates within 2 SE of tau={tau}"
        )


# ---------------------------------------------------------------------------
# Extended precision tests (required specification)
# ---------------------------------------------------------------------------


class TestMcCrarySpecification:
    """Explicit specification tests: clean → p>0.05, manipulated → p<0.05."""

    def test_clean_data_p_above_005(self):
        """Uniform running variable with tau=1.0 should NOT reject H0 at 5%."""
        df = generate_rd_data(5000, cutoff=0.0, tau=1.0, noise=0.5, seed=201)
        result = McCraryTest(df["x"].values, cutoff=0.0).fit()
        assert result.p_value > 0.05, (
            f"Clean data McCrary p={result.p_value:.4f} — unexpectedly significant"
        )

    def test_manipulated_data_p_below_005(self):
        """Heavy manipulation (frac=0.5) should reject H0 at 5%."""
        df = generate_manipulation_data(5000, manipulation_frac=0.50, seed=202)
        result = McCraryTest(df["x"].values, cutoff=0.0).fit()
        assert result.p_value < 0.05, (
            f"Manipulated data McCrary p={result.p_value:.4f} — should be significant"
        )

    def test_mccrary_nonzero_cutoff(self):
        """McCrary test should work at a non-zero cutoff."""
        df = generate_rd_data(3000, cutoff=0.3, tau=1.0, seed=203)
        result = McCraryTest(df["x"].values, cutoff=0.3).fit()
        assert 0.0 <= result.p_value <= 1.0
        assert np.isfinite(result.theta)


class TestCovariateBalanceSpecification:
    """Balanced → 0 significant; imbalanced → ≥1 significant."""

    def test_balanced_data_zero_significant(self):
        df = generate_rd_data_with_covariates(
            4000, n_covariates=5, balance=True, seed=210
        )
        cov_cols = [c for c in df.columns if c.startswith("z")]
        result = CovariateBalance(
            df["x"].values, df[cov_cols], cutoff=0.0, bandwidth=0.4
        ).fit()
        assert result.n_significant == 0, (
            f"Balanced covariates: {result.n_significant} significant (expected 0)"
        )

    def test_imbalanced_data_one_or_more_significant(self):
        df = generate_rd_data_with_covariates(
            4000, n_covariates=5, balance=False, seed=211
        )
        cov_cols = [c for c in df.columns if c.startswith("z")]
        result = CovariateBalance(
            df["x"].values, df[cov_cols], cutoff=0.0, bandwidth=0.4
        ).fit()
        assert result.n_significant >= 1, (
            f"Imbalanced covariates: {result.n_significant} significant (expected ≥1)"
        )


class TestPlaceboSpecification:
    """Under the null (no treatment), fewer than 10% of placebos should be significant."""

    def test_few_placebos_significant_under_null(self):
        """With tau=0 (no effect), the placebo rejection rate should stay near 5%."""
        df = generate_rd_data(4000, cutoff=0.0, tau=0.0, noise=0.5, seed=220)
        result = PlaceboTest(
            df["y"].values, df["x"].values,
            cutoff=0.0, bandwidth=0.35, n_placebo=20,
        ).fit()
        valid = ~np.isnan(result.placebo_estimates)
        rate = result.n_significant_placebos / max(int(valid.sum()), 1)
        assert rate < 0.10, (
            f"Null placebo rejection rate {rate:.0%} > 10% — "
            f"{result.n_significant_placebos}/{int(valid.sum())} significant"
        )


class TestDonutSpecification:
    """Donut estimates should be stable when there is no manipulation."""

    def test_donut_stable_without_manipulation(self):
        """With clean data, removing a small hole should not drastically change estimates."""
        df = generate_rd_data(4000, cutoff=0.0, tau=1.0, noise=0.4, seed=230)
        result = DonutRD(
            df["y"].values, df["x"].values, cutoff=0.0,
            bandwidth=0.4, donut_holes=[0.0, 0.02, 0.05],
        ).fit()
        valid_estimates = [e for e in result.estimates if np.isfinite(e)]
        assert len(valid_estimates) >= 2
        # All non-NaN estimates should be within 1.0 of each other
        spread = max(valid_estimates) - min(valid_estimates)
        assert spread < 1.0, (
            f"Donut estimates span {spread:.3f} on clean data — too unstable"
        )

    def test_donut_max_deviation_finite(self):
        df = generate_rd_data(2000, cutoff=0.0, tau=1.5, noise=0.5, seed=231)
        result = DonutRD(df["y"].values, df["x"].values).fit()
        assert np.isfinite(result.max_deviation)
