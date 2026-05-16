"""RD Credibility Score: a composite metric for RD design validity."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult, BandwidthSensitivity
from rd_credibility.diagnostics.covariate_balance import CovariateBalance, CovariateBalanceResult
from rd_credibility.diagnostics.mccrary import McCraryResult, McCraryTest
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult, PlaceboTest
from rd_credibility.estimation.rdrobust import RDEstimator


@dataclass
class CredibilityReport:
    """
    Comprehensive report from the RD credibility scoring system.

    Attributes
    ----------
    total_score : float
        Composite credibility score on [0, 100].
    grade : str
        Letter grade: A (85–100), B (70–84), C (55–69), D (40–54), F (<40).
    component_scores : dict
        Individual scores keyed by component name.
    component_explanations : dict
        One-line explanation for each component score.
    summary : str
        Auto-generated paragraph summarising the assessment.
    warnings : list of str
        Hard failure messages (e.g., manipulation detected).
    recommendations : list of str
        Actionable suggestions for improving design credibility.
    """

    total_score: float
    grade: str
    component_scores: dict
    component_explanations: dict
    summary: str
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


# Grade thresholds are opinionated but principled:
# A (85–100): All diagnostics pass comfortably. Design is publication-ready.
# B (70–84): Minor concerns in one dimension. Results are credible with caveats.
# C (55–69): Notable weakness in one or two diagnostics. Use caution.
# D (40–54): Serious concern. Results may not survive scrutiny.
# F (<40):   Fundamental validity threat. Design should not be relied upon.
_GRADE_THRESHOLDS = [
    (85, "A"),
    (70, "B"),
    (55, "C"),
    (40, "D"),
    (0, "F"),
]

_DEFAULT_WEIGHTS = {
    "manipulation": 0.25,
    "balance": 0.25,
    "sensitivity": 0.25,
    "placebo": 0.25,
}


def _assign_grade(score: float) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


class CredibilityScore:
    """
    Composite credibility score aggregating four RD diagnostic dimensions.

    Each dimension is scored independently (0–25 by default) and then
    summed.  Custom weights rescale each component proportionally so that
    the maximum total remains 100.

    Parameters
    ----------
    mccrary_result : McCraryResult
        Result of the McCrary density test.
    balance_result : CovariateBalanceResult
        Result of the covariate balance test.
    bandwidth_result : BandwidthGridResult
        Result of the bandwidth sensitivity analysis.
    placebo_result : PlaceboResult
        Result of the placebo cutoff test.
    weights : dict or None, optional
        Component weights keyed by name.  Values need not sum to 1 — they
        are normalised internally.  Default equal weights.
    """

    def __init__(
        self,
        mccrary_result: McCraryResult,
        balance_result: CovariateBalanceResult,
        bandwidth_result: BandwidthGridResult,
        placebo_result: PlaceboResult,
        weights: dict = None,
    ):
        self.mccrary_result = mccrary_result
        self.balance_result = balance_result
        self.bandwidth_result = bandwidth_result
        self.placebo_result = placebo_result
        self.weights = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)

        # Normalise weights to sum to 1
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute(self) -> CredibilityReport:
        """
        Compute the composite credibility score and generate the report.

        Returns
        -------
        CredibilityReport
        """
        manip_score, manip_expl = self._score_manipulation(self.mccrary_result)
        bal_score, bal_expl = self._score_balance(self.balance_result)
        sens_score, sens_expl = self._score_sensitivity(self.bandwidth_result)
        plac_score, plac_expl = self._score_placebo(self.placebo_result)

        raw_scores = {
            "manipulation": manip_score,
            "balance": bal_score,
            "sensitivity": sens_score,
            "placebo": plac_score,
        }

        # Weighted total (each raw score is on 0–25 scale; multiply by 4*weight
        # so that perfect = 100 regardless of weight distribution)
        total = sum(raw_scores[k] * 4.0 * self.weights[k] for k in raw_scores)
        total = max(0.0, min(100.0, total))

        # Hard-fail ceiling: when there is strong evidence of manipulation
        # (p < 0.01), no combination of passing diagnostics can rescue the
        # score above 25.  This encodes the principle that a compromised
        # assignment mechanism fundamentally invalidates the RD design.
        # For marginal evidence (0.01 <= p < 0.05), the zero component score
        # is punishment enough without capping the total.
        if raw_scores["manipulation"] == 0.0 and self.mccrary_result.p_value < 0.01:
            total = min(total, 25.0)

        grade = _assign_grade(total)

        component_scores = dict(raw_scores)
        component_explanations = {
            "manipulation": manip_expl,
            "balance": bal_expl,
            "sensitivity": sens_expl,
            "placebo": plac_expl,
        }

        warnings = self._generate_warnings(raw_scores)
        recommendations = self._generate_recommendations(raw_scores)
        summary = self._generate_summary(total, grade, component_explanations)

        return CredibilityReport(
            total_score=float(total),
            grade=grade,
            component_scores=component_scores,
            component_explanations=component_explanations,
            summary=summary,
            warnings=warnings,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Scoring rules
    # ------------------------------------------------------------------

    @staticmethod
    def _score_manipulation(result: McCraryResult) -> tuple:
        """
        Score the McCrary density test result.

        Parameters
        ----------
        result : McCraryResult

        Returns
        -------
        tuple of (float, str)
            (score on 0–25 scale, explanation)
        """
        p = result.p_value
        if p > 0.10:
            return 25.0, "No evidence of manipulation"
        elif p > 0.05:
            return 15.0, "Marginal evidence of manipulation"
        else:
            return 0.0, "Manipulation detected — design invalidated"

    @staticmethod
    def _score_balance(result: CovariateBalanceResult) -> tuple:
        """
        Score the covariate balance test result.

        Parameters
        ----------
        result : CovariateBalanceResult

        Returns
        -------
        tuple of (float, str)
            (score on 0–25 scale, explanation)
        """
        n_sig = result.n_significant
        df = result.results

        # Base score from significant count
        if n_sig == 0:
            score = 25.0
            expl = "All covariates balanced"
        elif n_sig == 1:
            score = 15.0
            expl = "1 covariate shows imbalance"
        elif n_sig == 2:
            score = 8.0
            expl = "2 covariates show imbalance"
        else:
            score = 0.0
            expl = f"{n_sig} covariates show imbalance — design suspect"

        # Deduct 2 points per covariate with p < 0.10 (beyond the base score penalty)
        n_marginal = int((df["p_value"].dropna() < 0.10).sum())
        deduction = 2.0 * n_marginal
        score = max(0.0, score - deduction)

        return score, expl

    @staticmethod
    def _score_sensitivity(result: BandwidthGridResult) -> tuple:
        """
        Score the bandwidth sensitivity analysis.

        Parameters
        ----------
        result : BandwidthGridResult

        Returns
        -------
        tuple of (float, str)
            (score on 0–25 scale, explanation)
        """
        cv = result.cv_of_estimates
        if np.isnan(cv):
            return 12.0, "Could not compute CV — insufficient variation"

        if cv < 0.10:
            return 25.0, "Highly stable across specifications"
        elif cv < 0.15:
            return 20.0, "Mostly stable"
        elif cv < 0.25:
            return 12.0, "Moderately sensitive to specification"
        else:
            return 5.0, "Highly sensitive to specification"

    @staticmethod
    def _score_placebo(result: PlaceboResult) -> tuple:
        """
        Score the placebo cutoff test.

        Parameters
        ----------
        result : PlaceboResult

        Returns
        -------
        tuple of (float, str)
            (score on 0–25 scale, explanation)
        """
        n_sig = result.n_significant_placebos

        if n_sig == 0:
            return 25.0, "No placebo effects detected"
        elif n_sig == 1:
            return 15.0, "1 placebo cutoff significant"
        elif n_sig == 2:
            return 5.0, "2 placebo cutoffs significant"
        else:
            return 0.0, f"{n_sig} placebo cutoffs significant — pervasive bias"

    # ------------------------------------------------------------------
    # Report generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_warnings(scores: dict) -> list:
        warnings = []
        if scores["manipulation"] == 0.0:
            warnings.append(
                "CRITICAL: McCrary test rejects continuity of the running "
                "variable density. Treatment effect estimates may be invalid."
            )
        if scores["balance"] == 0.0:
            warnings.append(
                "WARNING: Multiple covariates are discontinuous at the cutoff. "
                "The RD identifying assumption may be violated."
            )
        if scores["placebo"] == 0.0:
            warnings.append(
                "WARNING: Multiple placebo cutoffs show significant effects, "
                "suggesting systematic bias unrelated to treatment."
            )
        return warnings

    @staticmethod
    def _generate_recommendations(scores: dict) -> list:
        recs = []
        if scores["manipulation"] < 25.0:
            recs.append(
                "Investigate running variable bunching. Consider a donut-hole "
                "estimator excluding observations near the cutoff."
            )
        if scores["balance"] < 25.0:
            recs.append(
                "Include imbalanced covariates as controls or investigate "
                "whether they are affected by treatment."
            )
        if scores["sensitivity"] < 20.0:
            recs.append(
                "Report results across multiple bandwidths and polynomial "
                "orders. Consider using bias-corrected confidence intervals."
            )
        if scores["placebo"] < 25.0:
            recs.append(
                "Investigate why placebo cutoffs show effects. Check for "
                "other policy thresholds or functional form misspecification."
            )
        return recs

    @staticmethod
    def _generate_summary(total: float, grade: str, explanations: dict) -> str:
        parts = [
            f"Overall credibility score: {total:.1f}/100 (grade {grade}).",
            f"Manipulation: {explanations['manipulation']}.",
            f"Balance: {explanations['balance']}.",
            f"Sensitivity: {explanations['sensitivity']}.",
            f"Placebo: {explanations['placebo']}.",
        ]
        return " ".join(parts)


class CredibilityScoreCalibration:
    """
    Monte Carlo calibration of the credibility score.

    Runs many simulations with known-truth data to verify that the
    scoring system assigns high scores to valid designs and low scores
    to invalid ones.

    Parameters
    ----------
    n_simulations : int, optional
        Number of Monte Carlo replications. Default 50.
    n_obs : int, optional
        Observations per simulated dataset. Default 2000.
    true_tau : float, optional
        True treatment effect. Default 1.5.
    n_covariates : int, optional
        Number of pre-treatment covariates. Default 3.
    bandwidth : float, optional
        Common bandwidth for all diagnostics. Default 0.4.
    seed : int or None, optional
        Base random seed. Default None.
    """

    def __init__(
        self,
        n_simulations=50,
        n_obs=2000,
        true_tau=1.5,
        n_covariates=3,
        bandwidth=0.4,
        seed=None,
    ):
        self.n_simulations = int(n_simulations)
        self.n_obs = int(n_obs)
        self.true_tau = float(true_tau)
        self.n_covariates = int(n_covariates)
        self.bandwidth = float(bandwidth)
        self.seed = seed

    def simulate_and_score(self) -> pd.DataFrame:
        """
        Run Monte Carlo simulations and compute credibility scores.

        Returns
        -------
        pd.DataFrame
            One row per simulation with columns: sim_id, total_score, grade,
            manipulation_score, balance_score, sensitivity_score, placebo_score.
        """
        from tests.fixtures.synthetic_rd import (
            generate_rd_data_with_covariates,
        )

        rng = np.random.default_rng(self.seed)
        rows = []

        for i in range(self.n_simulations):
            sim_seed = int(rng.integers(0, 2**31))
            df = generate_rd_data_with_covariates(
                self.n_obs,
                cutoff=0.0,
                tau=self.true_tau,
                n_covariates=self.n_covariates,
                balance=True,
                seed=sim_seed,
            )

            y = df["y"].values
            x = df["x"].values
            cov_cols = [c for c in df.columns if c.startswith("z")]
            covariates = df[cov_cols]

            try:
                mccrary_res = McCraryTest(x, cutoff=0.0).fit()
                balance_res = CovariateBalance(
                    x, covariates, cutoff=0.0, bandwidth=self.bandwidth
                ).fit()
                bw_res = BandwidthSensitivity(
                    y, x, cutoff=0.0, poly_orders=[1, 2]
                ).fit()
                placebo_res = PlaceboTest(
                    y, x, cutoff=0.0, bandwidth=self.bandwidth, n_placebo=10
                ).fit()

                report = CredibilityScore(
                    mccrary_res, balance_res, bw_res, placebo_res
                ).compute()

                rows.append(
                    {
                        "sim_id": i,
                        "total_score": report.total_score,
                        "grade": report.grade,
                        "manipulation_score": report.component_scores["manipulation"],
                        "balance_score": report.component_scores["balance"],
                        "sensitivity_score": report.component_scores["sensitivity"],
                        "placebo_score": report.component_scores["placebo"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "sim_id": i,
                        "total_score": np.nan,
                        "grade": "ERR",
                        "manipulation_score": np.nan,
                        "balance_score": np.nan,
                        "sensitivity_score": np.nan,
                        "placebo_score": np.nan,
                    }
                )

        return pd.DataFrame(rows)

    def plot_distribution(self, results: pd.DataFrame = None):
        """
        Plot the distribution of calibration scores.

        Parameters
        ----------
        results : pd.DataFrame or None
            Output of simulate_and_score(). If None, runs the simulation first.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if results is None:
            results = self.simulate_and_score()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        scores = results["total_score"].dropna()
        axes[0].hist(scores, bins=15, edgecolor="black", alpha=0.7)
        axes[0].axvline(scores.mean(), color="red", linestyle="--", label=f"Mean={scores.mean():.1f}")
        axes[0].set_xlabel("Total Credibility Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Score Distribution (Valid Design)")
        axes[0].legend()

        components = ["manipulation_score", "balance_score", "sensitivity_score", "placebo_score"]
        means = [results[c].dropna().mean() for c in components]
        labels = ["Manipulation", "Balance", "Sensitivity", "Placebo"]
        axes[1].bar(labels, means, edgecolor="black", alpha=0.7)
        axes[1].set_ylabel("Mean Score (out of 25)")
        axes[1].set_title("Component Score Means")
        axes[1].set_ylim(0, 27)

        plt.tight_layout()
        return fig
