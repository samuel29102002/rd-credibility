"""Tests for the RD credibility scoring system."""

import numpy as np
import pandas as pd
import pytest

from rd_credibility.diagnostics.bandwidth_grid import BandwidthSensitivity
from rd_credibility.diagnostics.covariate_balance import CovariateBalance
from rd_credibility.diagnostics.mccrary import McCraryTest
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboTest
from rd_credibility.scoring.credibility import (
    CredibilityReport,
    CredibilityScore,
    CredibilityScoreCalibration,
)
from tests.fixtures.synthetic_rd import (
    generate_manipulation_data,
    generate_rd_data,
    generate_rd_data_with_covariates,
)


# ---------------------------------------------------------------------------
# Helpers: run all diagnostics on a dataset
# ---------------------------------------------------------------------------


def _run_diagnostics(df, n_covariates=3, bandwidth=0.4, cutoff=0.0):
    """Run all four diagnostics and return their results."""
    y = df["y"].values
    x = df["x"].values
    cov_cols = [c for c in df.columns if c.startswith("z")]
    if cov_cols:
        covariates = df[cov_cols]
    else:
        # Create dummy balanced covariates
        rng = np.random.default_rng(0)
        covariates = pd.DataFrame(
            {f"z{i}": rng.normal(size=len(x)) for i in range(n_covariates)}
        )

    mccrary_res = McCraryTest(x, cutoff=cutoff).fit()
    balance_res = CovariateBalance(x, covariates, cutoff=cutoff, bandwidth=bandwidth).fit()
    bw_res = BandwidthSensitivity(y, x, cutoff=cutoff, poly_orders=[1, 2]).fit()
    placebo_res = PlaceboTest(y, x, cutoff=cutoff, bandwidth=bandwidth, n_placebo=10).fit()

    return mccrary_res, balance_res, bw_res, placebo_res


# ---------------------------------------------------------------------------
# Individual scoring rules
# ---------------------------------------------------------------------------


class TestScoringRules:
    def test_manipulation_no_evidence(self):
        from rd_credibility.diagnostics.mccrary import McCraryResult

        result = McCraryResult(
            theta=0.1, se=0.5, t_stat=0.2, p_value=0.84,
            bin_centers=np.array([]), bin_counts=np.array([]),
            fitted_left=np.array([]), fitted_right=np.array([]),
            conclusion="No evidence of manipulation",
        )
        score, expl = CredibilityScore._score_manipulation(result)
        assert score == 25.0
        assert "No evidence" in expl

    def test_manipulation_marginal(self):
        from rd_credibility.diagnostics.mccrary import McCraryResult

        result = McCraryResult(
            theta=0.5, se=0.25, t_stat=2.0, p_value=0.07,
            bin_centers=np.array([]), bin_counts=np.array([]),
            fitted_left=np.array([]), fitted_right=np.array([]),
            conclusion="No evidence of manipulation",
        )
        score, _ = CredibilityScore._score_manipulation(result)
        assert score == 15.0

    def test_manipulation_detected(self):
        from rd_credibility.diagnostics.mccrary import McCraryResult

        result = McCraryResult(
            theta=2.0, se=0.5, t_stat=4.0, p_value=0.001,
            bin_centers=np.array([]), bin_counts=np.array([]),
            fitted_left=np.array([]), fitted_right=np.array([]),
            conclusion="Manipulation detected",
        )
        score, expl = CredibilityScore._score_manipulation(result)
        assert score == 0.0
        assert "invalidated" in expl

    def test_balance_all_balanced(self):
        from rd_credibility.diagnostics.covariate_balance import CovariateBalanceResult

        df = pd.DataFrame({
            "covariate": ["z0", "z1", "z2"],
            "estimate": [0.01, -0.02, 0.03],
            "se": [0.1, 0.1, 0.1],
            "p_value": [0.90, 0.80, 0.70],
            "significant": [False, False, False],
        })
        result = CovariateBalanceResult(results=df, n_significant=0,
                                        overall_conclusion="Balanced", mean_p_value=0.8)
        score, _ = CredibilityScore._score_balance(result)
        assert score == 25.0

    def test_balance_3_significant(self):
        from rd_credibility.diagnostics.covariate_balance import CovariateBalanceResult

        df = pd.DataFrame({
            "covariate": ["z0", "z1", "z2"],
            "estimate": [0.5, 0.6, 0.7],
            "se": [0.1, 0.1, 0.1],
            "p_value": [0.001, 0.002, 0.003],
            "significant": [True, True, True],
        })
        result = CovariateBalanceResult(results=df, n_significant=3,
                                        overall_conclusion="Imbalanced", mean_p_value=0.002)
        score, _ = CredibilityScore._score_balance(result)
        assert score == 0.0  # Base 0, deductions cannot go below 0

    def test_sensitivity_stable(self):
        from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult

        grid = pd.DataFrame({"bandwidth": [0.3], "poly_order": [1],
                             "estimate": [1.5], "se": [0.2],
                             "ci_lower": [1.1], "ci_upper": [1.9]})
        result = BandwidthGridResult(grid=grid, optimal_bandwidth=0.3,
                                     cv_of_estimates=0.05, stable_region=(0.2, 0.5))
        score, expl = CredibilityScore._score_sensitivity(result)
        assert score == 25.0
        assert "Highly stable" in expl

    def test_sensitivity_high_cv(self):
        from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult

        grid = pd.DataFrame({"bandwidth": [0.3], "poly_order": [1],
                             "estimate": [1.5], "se": [0.2],
                             "ci_lower": [1.1], "ci_upper": [1.9]})
        result = BandwidthGridResult(grid=grid, optimal_bandwidth=0.3,
                                     cv_of_estimates=0.30, stable_region=(0.2, 0.5))
        score, expl = CredibilityScore._score_sensitivity(result)
        assert score == 5.0
        assert "Highly sensitive" in expl

    def test_placebo_none_significant(self):
        from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult

        result = PlaceboResult(
            placebo_cutoffs=np.array([-0.5, -0.3, 0.3, 0.5]),
            placebo_estimates=np.array([0.1, -0.1, 0.05, -0.05]),
            placebo_ses=np.array([0.3, 0.3, 0.3, 0.3]),
            true_estimate=2.0, true_se=0.3,
            n_significant_placebos=0,
            conclusion="No placebos significant",
        )
        score, _ = CredibilityScore._score_placebo(result)
        assert score == 25.0

    def test_placebo_three_significant(self):
        from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult

        result = PlaceboResult(
            placebo_cutoffs=np.array([-0.5, -0.3, 0.3, 0.5]),
            placebo_estimates=np.array([1.0, 1.2, 0.9, 1.1]),
            placebo_ses=np.array([0.3, 0.3, 0.3, 0.3]),
            true_estimate=2.0, true_se=0.3,
            n_significant_placebos=3,
            conclusion="Design concern",
        )
        score, _ = CredibilityScore._score_placebo(result)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Integration: valid RD → high score
# ---------------------------------------------------------------------------


class TestIntegrationValidDesign:
    @pytest.fixture(scope="class")
    def valid_report(self):
        df = generate_rd_data_with_covariates(
            5000, cutoff=0.0, tau=2.0, n_covariates=3, balance=True, seed=100
        )
        results = _run_diagnostics(df, bandwidth=0.4)
        return CredibilityScore(*results).compute()

    def test_score_above_80(self, valid_report):
        assert valid_report.total_score > 80, (
            f"Valid design scored only {valid_report.total_score:.1f}"
        )

    def test_grade_a_or_b(self, valid_report):
        assert valid_report.grade in ("A", "B"), (
            f"Valid design got grade {valid_report.grade}"
        )

    def test_no_critical_warnings(self, valid_report):
        critical = [w for w in valid_report.warnings if "CRITICAL" in w]
        assert len(critical) == 0

    def test_report_has_summary(self, valid_report):
        assert isinstance(valid_report.summary, str)
        assert len(valid_report.summary) > 20

    def test_all_components_present(self, valid_report):
        for key in ("manipulation", "balance", "sensitivity", "placebo"):
            assert key in valid_report.component_scores
            assert key in valid_report.component_explanations


# ---------------------------------------------------------------------------
# Integration: manipulated data → low score
# ---------------------------------------------------------------------------


class TestIntegrationManipulated:
    @pytest.fixture(scope="class")
    def manipulated_report(self):
        df = generate_manipulation_data(
            10000, cutoff=0.0, manipulation_frac=0.90, seed=200
        )
        # Add dummy covariates
        rng = np.random.default_rng(200)
        for i in range(3):
            df[f"z{i}"] = rng.normal(size=len(df))
        results = _run_diagnostics(df, bandwidth=0.4)
        return CredibilityScore(*results).compute()

    def test_score_below_40(self, manipulated_report):
        assert manipulated_report.total_score < 40, (
            f"Manipulated design scored {manipulated_report.total_score:.1f} — should be < 40"
        )

    def test_grade_f(self, manipulated_report):
        assert manipulated_report.grade == "F", (
            f"Manipulated design got grade {manipulated_report.grade}"
        )

    def test_has_critical_warning(self, manipulated_report):
        assert any("CRITICAL" in w for w in manipulated_report.warnings)

    def test_manipulation_score_zero(self, manipulated_report):
        assert manipulated_report.component_scores["manipulation"] == 0.0


# ---------------------------------------------------------------------------
# Integration: imbalanced covariates → medium-low score
# ---------------------------------------------------------------------------


class TestIntegrationImbalanced:
    @pytest.fixture(scope="class")
    def imbalanced_report(self):
        df = generate_rd_data_with_covariates(
            5000, cutoff=0.0, tau=1.5, n_covariates=5, balance=False, seed=300
        )
        results = _run_diagnostics(df, bandwidth=0.4)
        return CredibilityScore(*results).compute()

    def test_score_below_60(self, imbalanced_report):
        assert imbalanced_report.total_score < 60, (
            f"Imbalanced design scored {imbalanced_report.total_score:.1f} — should be < 60"
        )

    def test_balance_score_low(self, imbalanced_report):
        assert imbalanced_report.component_scores["balance"] < 15.0

    def test_has_balance_recommendation(self, imbalanced_report):
        assert any("covariate" in r.lower() for r in imbalanced_report.recommendations)


# ---------------------------------------------------------------------------
# CredibilityReport structure
# ---------------------------------------------------------------------------


class TestCredibilityReport:
    def test_grade_boundaries(self):
        from rd_credibility.scoring.credibility import _assign_grade

        assert _assign_grade(100) == "A"
        assert _assign_grade(85) == "A"
        assert _assign_grade(84.9) == "B"
        assert _assign_grade(70) == "B"
        assert _assign_grade(69.9) == "C"
        assert _assign_grade(55) == "C"
        assert _assign_grade(54.9) == "D"
        assert _assign_grade(40) == "D"
        assert _assign_grade(39.9) == "F"
        assert _assign_grade(0) == "F"

    def test_custom_weights(self):
        """Custom weights should still produce a valid 0–100 score."""
        df = generate_rd_data_with_covariates(
            2000, cutoff=0.0, tau=2.0, n_covariates=2, balance=True, seed=400
        )
        results = _run_diagnostics(df, n_covariates=2, bandwidth=0.4)
        custom_weights = {"manipulation": 0.5, "balance": 0.2, "sensitivity": 0.2, "placebo": 0.1}
        report = CredibilityScore(*results, weights=custom_weights).compute()
        assert 0.0 <= report.total_score <= 100.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_simulate_and_score_returns_dataframe(self):
        cal = CredibilityScoreCalibration(
            n_simulations=3, n_obs=1000, true_tau=1.5, seed=500
        )
        results = cal.simulate_and_score()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        expected_cols = {"sim_id", "total_score", "grade",
                         "manipulation_score", "balance_score",
                         "sensitivity_score", "placebo_score"}
        assert set(results.columns) >= expected_cols

    def test_valid_design_calibration_scores_high(self):
        """With n_simulations valid datasets, the median score should be > 70."""
        cal = CredibilityScoreCalibration(
            n_simulations=5, n_obs=5000, true_tau=1.5, n_covariates=2, seed=600
        )
        results = cal.simulate_and_score()
        valid_scores = results["total_score"].dropna()
        assert valid_scores.median() > 70, (
            f"Calibration median={valid_scores.median():.1f}, expected > 70"
        )


# ---------------------------------------------------------------------------
# Monte Carlo tests (required specification)
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    """
    Run multiple synthetic datasets and verify aggregate scoring behaviour.
    These are slower tests but provide the strongest statistical guarantees.
    """

    @pytest.mark.slow
    def test_clean_designs_mean_above_75(self):
        """100 clean RDs should average > 75 on the credibility scale."""
        scores = []
        for seed in range(100):
            df = generate_rd_data_with_covariates(
                2000, tau=1.0, n_covariates=3, balance=True, seed=seed
            )
            results = _run_diagnostics(df, bandwidth=0.4)
            report = CredibilityScore(*results).compute()
            scores.append(report.total_score)
        mean_score = float(np.mean(scores))
        assert mean_score > 75, (
            f"Monte Carlo clean-design mean={mean_score:.1f} — expected > 75"
        )

    @pytest.mark.slow
    def test_manipulated_designs_mean_below_45(self):
        """100 manipulated RDs should average < 45 on the credibility scale."""
        scores = []
        for seed in range(100):
            df = generate_manipulation_data(
                3000, manipulation_frac=0.40, seed=seed
            )
            rng = np.random.default_rng(seed)
            for i in range(3):
                df[f"z{i}"] = rng.normal(size=len(df))
            results = _run_diagnostics(df, bandwidth=0.4)
            report = CredibilityScore(*results).compute()
            scores.append(report.total_score)
        mean_score = float(np.mean(scores))
        assert mean_score < 45, (
            f"Monte Carlo manipulated mean={mean_score:.1f} — expected < 45"
        )

    def test_score_decreases_with_manipulation_severity(self):
        """
        As manipulation_frac increases from 0.0 → 0.6, the median credibility
        score should show a downward trend.
        """
        fracs = [0.0, 0.15, 0.30, 0.45, 0.60]
        median_scores = []
        for frac in fracs:
            rep_scores = []
            for seed in range(10):
                if frac == 0.0:
                    df = generate_rd_data(2000, tau=1.0, noise=0.5, seed=seed + 500)
                    rng = np.random.default_rng(seed + 500)
                    for i in range(3):
                        df[f"z{i}"] = rng.normal(size=len(df))
                else:
                    df = generate_manipulation_data(2000, manipulation_frac=frac, seed=seed + 500)
                    rng = np.random.default_rng(seed + 500)
                    for i in range(3):
                        df[f"z{i}"] = rng.normal(size=len(df))
                results = _run_diagnostics(df, bandwidth=0.4)
                rep = CredibilityScore(*results).compute()
                rep_scores.append(rep.total_score)
            median_scores.append(float(np.median(rep_scores)))

        # The score sequence should be non-increasing overall
        # (allow one "blip" due to sampling noise, but require overall downward trend)
        first = median_scores[0]
        last = median_scores[-1]
        assert first > last, (
            f"Score not lower at high manipulation: {first:.1f} → {last:.1f}\n"
            f"Full sequence: {[f'{s:.1f}' for s in median_scores]}"
        )

    def test_grade_always_valid(self):
        """CredibilityReport.grade must always be in {{A,B,C,D,F}}."""
        valid_grades = {"A", "B", "C", "D", "F"}
        for seed in range(20):
            df = generate_rd_data_with_covariates(
                1000, tau=1.0, n_covariates=2, balance=True, seed=seed + 700
            )
            results = _run_diagnostics(df, bandwidth=0.4)
            report = CredibilityScore(*results).compute()
            assert report.grade in valid_grades, (
                f"Seed {seed}: invalid grade '{report.grade}'"
            )


# ---------------------------------------------------------------------------
# Scoring rule edge cases (cover uncovered branches)
# ---------------------------------------------------------------------------


class TestScoringEdgeCases:
    def test_balance_one_significant(self):
        """n_significant=1 → score 15, then marginal deduction."""
        from rd_credibility.diagnostics.covariate_balance import CovariateBalanceResult
        df = pd.DataFrame({
            "covariate": ["z0", "z1", "z2"],
            "estimate": [0.5, 0.1, -0.1],
            "se": [0.2, 0.15, 0.12],
            "p_value": [0.01, 0.40, 0.55],
            "significant": [True, False, False],
        })
        result = CovariateBalanceResult(
            results=df, n_significant=1,
            overall_conclusion="Marginal imbalance", mean_p_value=0.32,
        )
        score, expl = CredibilityScore._score_balance(result)
        assert score > 0.0  # 15 base minus deductions, but base > 0
        assert "1 covariate" in expl

    def test_balance_two_significant(self):
        from rd_credibility.diagnostics.covariate_balance import CovariateBalanceResult
        df = pd.DataFrame({
            "covariate": ["z0", "z1", "z2"],
            "estimate": [0.5, 0.6, 0.1],
            "se": [0.15, 0.14, 0.12],
            "p_value": [0.005, 0.008, 0.50],
            "significant": [True, True, False],
        })
        result = CovariateBalanceResult(
            results=df, n_significant=2,
            overall_conclusion="Imbalanced", mean_p_value=0.17,
        )
        score, expl = CredibilityScore._score_balance(result)
        assert "2 covariates" in expl

    def test_sensitivity_nan_cv(self):
        """CV=NaN (degenerate grid) → score 12 (midpoint fallback)."""
        from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult
        grid = pd.DataFrame({
            "bandwidth": [0.3], "poly_order": [1],
            "estimate": [float("nan")], "se": [float("nan")],
            "ci_lower": [float("nan")], "ci_upper": [float("nan")],
        })
        result = BandwidthGridResult(
            grid=grid, optimal_bandwidth=0.3,
            cv_of_estimates=float("nan"), stable_region=(0.2, 0.5),
        )
        score, expl = CredibilityScore._score_sensitivity(result)
        assert score == 12.0
        assert "Could not compute" in expl

    def test_placebo_one_significant(self):
        """n_significant_placebos=1 → score 15."""
        from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult
        result = PlaceboResult(
            placebo_cutoffs=np.array([-0.5, -0.3, 0.3, 0.5]),
            placebo_estimates=np.array([1.1, 0.1, 0.0, -0.1]),
            placebo_ses=np.array([0.3, 0.3, 0.3, 0.3]),
            true_estimate=2.0, true_se=0.3,
            n_significant_placebos=1,
            conclusion="One placebo significant",
        )
        score, _ = CredibilityScore._score_placebo(result)
        assert score == 15.0

    def test_placebo_two_significant(self):
        """n_significant_placebos=2 → score 5."""
        from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult
        result = PlaceboResult(
            placebo_cutoffs=np.array([-0.5, -0.3, 0.3, 0.5]),
            placebo_estimates=np.array([1.1, 1.2, 0.0, -0.1]),
            placebo_ses=np.array([0.3, 0.3, 0.3, 0.3]),
            true_estimate=2.0, true_se=0.3,
            n_significant_placebos=2,
            conclusion="Two placebos significant",
        )
        score, _ = CredibilityScore._score_placebo(result)
        assert score == 5.0

    def test_sensitivity_cv_bounds(self):
        """Test all four CV bands of sensitivity scoring."""
        from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult
        dummy_grid = pd.DataFrame({
            "bandwidth": [0.3], "poly_order": [1],
            "estimate": [1.5], "se": [0.2],
            "ci_lower": [1.1], "ci_upper": [1.9],
        })

        for cv, expected_score in [(0.05, 25.0), (0.12, 20.0), (0.20, 12.0), (0.30, 5.0)]:
            result = BandwidthGridResult(
                grid=dummy_grid, optimal_bandwidth=0.3,
                cv_of_estimates=cv, stable_region=(0.2, 0.5),
            )
            score, _ = CredibilityScore._score_sensitivity(result)
            assert score == expected_score, f"CV={cv}: expected {expected_score}, got {score}"
