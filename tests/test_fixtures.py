"""Tests for the synthetic RD data generators."""

import numpy as np
import pytest

from tests.fixtures.synthetic_rd import (
    generate_manipulation_data,
    generate_rd_data,
    generate_rd_data_with_covariates,
)

# ---------------------------------------------------------------------------
# generate_rd_data
# ---------------------------------------------------------------------------


class TestGenerateRdData:
    REQUIRED_COLS = {"x", "d", "y", "y0", "y1"}

    def test_columns(self):
        df = generate_rd_data(100, seed=0)
        assert set(df.columns) == self.REQUIRED_COLS

    def test_no_nans(self):
        df = generate_rd_data(500, seed=1)
        assert df.isna().sum().sum() == 0

    def test_sharp_assignment(self):
        df = generate_rd_data(2000, cutoff=0.0, seed=2)
        assert (df.loc[df["x"] >= 0.0, "d"] == 1).all()
        assert (df.loc[df["x"] < 0.0, "d"] == 0).all()

    def test_x_in_valid_range(self):
        df = generate_rd_data(1000, seed=3)
        assert df["x"].between(-1.0, 1.0).all()

    def test_potential_outcomes_differ_by_tau(self):
        tau = 2.5
        df = generate_rd_data(500, tau=tau, seed=4)
        # y1 - y0 equals tau everywhere; noise enters both identically
        np.testing.assert_allclose(df["y1"] - df["y0"], tau)

    def test_observed_outcome_matches_assignment(self):
        df = generate_rd_data(500, tau=1.0, seed=5)
        np.testing.assert_array_equal(df.loc[df["d"] == 1, "y"], df.loc[df["d"] == 1, "y1"])
        np.testing.assert_array_equal(df.loc[df["d"] == 0, "y"], df.loc[df["d"] == 0, "y0"])

    def test_reproducible_with_seed(self):
        df1 = generate_rd_data(200, seed=99)
        df2 = generate_rd_data(200, seed=99)
        assert df1.equals(df2)

    def test_ate_approximates_tau_large_n(self):
        """Simple local mean difference near the cutoff should recover tau."""
        tau = 1.5
        cutoff = 0.0
        bandwidth = 0.05

        df = generate_rd_data(
            50_000,
            cutoff=cutoff,
            tau=tau,
            slope_left=1.0,
            slope_right=1.0,
            noise=0.3,
            seed=0,
        )

        near = df[np.abs(df["x"] - cutoff) < bandwidth]
        estimate = near.loc[near["d"] == 1, "y"].mean() - near.loc[near["d"] == 0, "y"].mean()

        assert abs(estimate - tau) < 0.2, (
            f"Local mean difference {estimate:.3f} deviates too far from tau={tau}"
        )


# ---------------------------------------------------------------------------
# generate_rd_data_with_covariates
# ---------------------------------------------------------------------------


class TestGenerateRdDataWithCovariates:
    BASE_COLS = {"x", "d", "y", "y0", "y1"}

    def test_columns_balanced(self):
        df = generate_rd_data_with_covariates(100, n_covariates=3, balance=True, seed=0)
        assert set(df.columns) == self.BASE_COLS | {"z0", "z1", "z2"}

    def test_columns_imbalanced(self):
        df = generate_rd_data_with_covariates(100, n_covariates=2, balance=False, seed=0)
        assert set(df.columns) == self.BASE_COLS | {"z0", "z1"}

    def test_no_nans(self):
        df = generate_rd_data_with_covariates(300, n_covariates=4, seed=1)
        assert df.isna().sum().sum() == 0

    def test_balanced_covariates_no_jump(self):
        """Balanced covariates must not show a meaningful mean shift at the cutoff."""
        df = generate_rd_data_with_covariates(
            20_000, cutoff=0.0, n_covariates=1, balance=True, seed=7
        )
        left_mean = df.loc[df["d"] == 0, "z0"].mean()
        right_mean = df.loc[df["d"] == 1, "z0"].mean()
        assert abs(right_mean - left_mean) < 0.1, (
            f"Balanced covariate jump ({right_mean - left_mean:.3f}) should be near 0"
        )

    def test_imbalanced_covariates_have_jump(self):
        """Imbalanced covariates must show a clear mean shift at the cutoff."""
        df = generate_rd_data_with_covariates(
            10_000, cutoff=0.0, n_covariates=1, balance=False, seed=8
        )
        left_mean = df.loc[df["d"] == 0, "z0"].mean()
        right_mean = df.loc[df["d"] == 1, "z0"].mean()
        assert abs(right_mean - left_mean) > 0.3, (
            f"Imbalanced covariate jump ({right_mean - left_mean:.3f}) should be substantial"
        )

    def test_zero_covariates(self):
        df = generate_rd_data_with_covariates(100, n_covariates=0, seed=0)
        assert set(df.columns) == self.BASE_COLS


# ---------------------------------------------------------------------------
# generate_manipulation_data
# ---------------------------------------------------------------------------


class TestGenerateManipulationData:
    REQUIRED_COLS = {"x", "d", "y", "y0", "y1", "manipulated"}

    def test_columns(self):
        df = generate_manipulation_data(100, seed=0)
        assert set(df.columns) == self.REQUIRED_COLS

    def test_no_nans(self):
        df = generate_manipulation_data(300, seed=1)
        assert df.isna().sum().sum() == 0

    def test_manipulated_is_boolean(self):
        df = generate_manipulation_data(200, seed=2)
        assert df["manipulated"].dtype == bool

    def test_zero_manipulation_no_flags(self):
        df = generate_manipulation_data(500, manipulation_frac=0.0, seed=3)
        assert df["manipulated"].sum() == 0

    def test_manipulated_units_above_cutoff(self):
        """All flagged units must end up above (or at) the cutoff after shifting."""
        cutoff = 0.0
        df = generate_manipulation_data(1000, cutoff=cutoff, manipulation_frac=0.5, seed=4)
        assert (df.loc[df["manipulated"], "x"] >= cutoff).all()

    def test_manipulation_flag_count_approx(self):
        """Number of manipulated units should be close to the expected count."""
        n = 20_000
        cutoff = 0.0
        bandwidth = 0.2
        manipulation_frac = 0.5

        df = generate_manipulation_data(
            n, cutoff=cutoff, manipulation_frac=manipulation_frac, seed=5
        )

        # Expected near-below count: n * bandwidth / total_range * 0.5 of the range
        expected_near_below = n * bandwidth / 2.0  # fraction of Uniform(-1,1) in window
        expected_manipulated = manipulation_frac * expected_near_below

        actual = df["manipulated"].sum()
        relative_error = abs(actual - expected_manipulated) / expected_manipulated
        assert relative_error < 0.3, (
            f"Manipulated count {actual} too far from expected {expected_manipulated:.0f}"
        )

    def test_density_check_fails_with_manipulation(self):
        """Manipulation creates excess density just above the cutoff."""
        cutoff = 0.0
        window = 0.05

        df_clean = generate_manipulation_data(
            20_000, cutoff=cutoff, manipulation_frac=0.0, seed=10
        )
        df_manip = generate_manipulation_data(
            20_000, cutoff=cutoff, manipulation_frac=0.9, seed=10
        )

        def density_ratio(df):
            above = ((df["x"] >= cutoff) & (df["x"] < cutoff + window)).sum()
            below = ((df["x"] >= cutoff - window) & (df["x"] < cutoff)).sum()
            return above / (below + 1e-10)

        ratio_clean = density_ratio(df_clean)
        ratio_manip = density_ratio(df_manip)

        assert abs(ratio_clean - 1.0) < 0.4, (
            f"Clean density ratio {ratio_clean:.2f} should be near 1"
        )
        assert ratio_manip > ratio_clean + 0.3, (
            f"Manipulation ratio {ratio_manip:.2f} should substantially exceed "
            f"clean ratio {ratio_clean:.2f}"
        )
