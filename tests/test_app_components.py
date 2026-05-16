"""Tests for app components: data_loader and replication engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rd_credibility.app.components.data_loader import (
    DEFAULT_CUTOFFS,
    _lee_2008,
    _maimonides,
    _thistlethwaite,
    covariate_columns,
    load_builtin,
)
from rd_credibility.app.components.replication import (
    PreloadedExample,
    ReplicationAudit,
    _determine_verdict,
    get_preloaded_examples,
    run_replication_audit,
)


# ---------------------------------------------------------------------------
# data_loader
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_lee_returns_dataframe(self):
        df = _lee_2008(seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_lee_required_columns(self):
        df = _lee_2008(seed=0)
        assert {"x", "y", "d"}.issubset(df.columns)

    def test_maimonides_returns_dataframe(self):
        df = _maimonides(seed=1)
        assert isinstance(df, pd.DataFrame)
        assert {"x", "y", "d"}.issubset(df.columns)

    def test_thistlethwaite_returns_dataframe(self):
        df = _thistlethwaite(seed=2)
        assert isinstance(df, pd.DataFrame)
        assert {"x", "y", "d"}.issubset(df.columns)

    def test_lee_y_bounded(self):
        df = _lee_2008(seed=0)
        assert df["y"].between(0, 1).all()

    def test_lee_treatment_sharp(self):
        df = _lee_2008(seed=0)
        expected_d = (df["x"] >= 0).astype(float)
        assert (df["d"] == expected_d).all()

    def test_covariate_columns_lee(self):
        df = _lee_2008(seed=0)
        covs = covariate_columns(df)
        assert len(covs) >= 1
        assert all(c.startswith("z") for c in covs)

    def test_load_builtin_all_datasets(self):
        for name in ["Lee 2008 (Electoral)", "Maimonides Rule", "Thistlethwaite 1960"]:
            df = load_builtin(name)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 100

    def test_load_builtin_unknown_raises(self):
        with pytest.raises(KeyError):
            load_builtin("Unknown Dataset")

    def test_default_cutoffs_all_zero(self):
        for name, c in DEFAULT_CUTOFFS.items():
            assert c == 0.0, f"{name}: expected cutoff=0.0, got {c}"

    def test_datasets_have_covariates(self):
        for name in ["Lee 2008 (Electoral)", "Maimonides Rule", "Thistlethwaite 1960"]:
            df = load_builtin(name)
            assert len(covariate_columns(df)) >= 1, f"{name} has no covariates"

    def test_datasets_reproducible(self):
        """Same seed should produce same data."""
        df1 = _lee_2008(seed=42)
        df2 = _lee_2008(seed=42)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# replication engine — unit tests
# ---------------------------------------------------------------------------


class TestReplicationPreloadedExamples:
    @pytest.fixture(scope="class")
    def examples(self):
        return get_preloaded_examples()

    def test_all_three_examples_built(self, examples):
        assert len(examples) == 3

    def test_example_names(self, examples):
        names = set(examples.keys())
        assert "Lee 2008 (Electoral)" in names
        assert "Synthetic: Manipulated Design" in names
        assert "Synthetic: Fragile Design" in names

    def test_each_example_has_required_fields(self, examples):
        for name, ex in examples.items():
            assert isinstance(ex.df, pd.DataFrame), f"{name}: df not a DataFrame"
            assert ex.reported_bandwidth > 0, f"{name}: bandwidth <= 0"
            assert np.isfinite(ex.reported_estimate), f"{name}: estimate not finite"
            assert ex.reported_se > 0, f"{name}: SE <= 0"
            assert ex.cutoff == 0.0, f"{name}: expected cutoff=0.0"

    def test_expected_verdicts_match(self, examples):
        verdicts = {name: ex.expected_verdict for name, ex in examples.items()}
        assert verdicts["Lee 2008 (Electoral)"] == "Robust"
        assert verdicts["Synthetic: Manipulated Design"] == "Problematic"
        assert verdicts["Synthetic: Fragile Design"] == "Fragile"


class TestRunReplicationAudit:
    @pytest.fixture(scope="class")
    def lee_audit(self):
        from rd_credibility.app.components.data_loader import _lee_2008
        from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
        from rd_credibility.estimation.rdrobust import RDEstimator

        df = _lee_2008(seed=42)
        y, x = df["y"].values, df["x"].values
        bw = mse_optimal_bandwidth(y, x, 0.0)
        rd = RDEstimator(y, x, cutoff=0.0, bandwidth=bw).fit()
        cov_cols = [c for c in df.columns if c.startswith("z")]
        cov_data = df[cov_cols].values
        return run_replication_audit(
            y, x, cov_cols, cov_data,
            cutoff=0.0,
            reported_bandwidth=bw,
            reported_estimate=rd.estimate,
            reported_se=rd.se,
        )

    def test_audit_returns_replication_audit(self, lee_audit):
        assert isinstance(lee_audit, ReplicationAudit)

    def test_lee_verdict_robust(self, lee_audit):
        assert lee_audit.verdict == "Robust"

    def test_estimate_audit_fields(self, lee_audit):
        ea = lee_audit.estimate
        assert np.isfinite(ea.our_estimate)
        assert ea.our_se > 0
        assert np.isfinite(ea.diff_in_se) and ea.diff_in_se >= 0
        assert ea.status in ("reproduced", "marginal", "failed")

    def test_bandwidth_audit_fields(self, lee_audit):
        ba = lee_audit.bandwidth
        assert ba.reported_bw > 0
        assert ba.mse_optimal_bw > 0
        assert ba.ratio > 0
        assert isinstance(ba.in_stable_region, bool)
        lo, hi = ba.stable_region
        assert lo <= hi

    def test_fragility_audit_fields(self, lee_audit):
        fa = lee_audit.fragility
        assert 0.0 <= fa.fragility_score <= 1.0
        assert fa.n_nearby >= 0
        assert fa.status in ("stable", "moderate", "fragile")

    def test_diagnostic_audit_fields(self, lee_audit):
        da = lee_audit.diagnostics
        assert da.mccrary_status in ("pass", "warn", "fail")
        assert da.balance_status in ("pass", "warn", "fail")
        assert da.placebo_status in ("pass", "warn", "fail")
        assert da.has_covariates is True

    def test_credibility_score_valid(self, lee_audit):
        assert 0 <= lee_audit.credibility_score <= 100
        assert lee_audit.credibility_grade in ("A", "B", "C", "D", "F")

    def test_no_covariates_runs_cleanly(self):
        """Audit should not crash when cov_data=None."""
        from tests.fixtures.synthetic_rd import generate_rd_data
        df = generate_rd_data(1000, tau=1.0, seed=99)
        y, x = df["y"].values, df["x"].values
        audit = run_replication_audit(
            y, x, cov_cols=[], cov_data=None,
            cutoff=0.0, reported_bandwidth=0.4,
            reported_estimate=1.0, reported_se=0.15,
        )
        assert isinstance(audit, ReplicationAudit)
        assert audit.diagnostics.has_covariates is False


class TestDetermineVerdict:
    """Unit-test _determine_verdict without running the full pipeline."""

    def _make_audits(self, *, reproduced=True, diff_in_se=0.5,
                     in_stable=True, bw_ratio=1.0, frag_status="stable",
                     frag_score=0.90, mccrary_status="pass",
                     balance_status="pass", placebo_status="pass"):
        from rd_credibility.app.components.replication import (
            EstimateAudit, BandwidthAudit, FragilityAudit, DiagnosticAudit,
        )
        from unittest.mock import MagicMock

        est = EstimateAudit(
            our_estimate=1.0, our_se=0.1,
            reported_estimate=1.0, reported_se=0.1,
            diff=0.0, diff_in_se=diff_in_se,
            reproduced=reproduced, status="reproduced" if reproduced else "failed",
        )
        bw = BandwidthAudit(
            reported_bw=0.4, mse_optimal_bw=0.4 / bw_ratio if bw_ratio else 0.4,
            ratio=bw_ratio, in_stable_region=in_stable,
            stable_region=(0.3, 0.5), status="optimal" if in_stable else "suspicious",
        )
        frag = FragilityAudit(
            fragility_score=frag_score, n_nearby=10, n_stable=int(frag_score * 10),
            status=frag_status,
        )
        mccrary_mock = MagicMock()
        mccrary_mock.p_value = 0.50
        mccrary_mock.conclusion = "No evidence"
        mccrary_mock.theta = 0.1
        mccrary_mock.se = 0.2
        mccrary_mock.t_stat = 0.5

        placebo_mock = MagicMock()
        placebo_mock.n_significant_placebos = 0
        placebo_mock.placebo_estimates = np.zeros(10)

        balance_mock = MagicMock()
        balance_mock.n_significant = 0
        balance_mock.results = pd.DataFrame({"p_value": [0.5]})

        diag = DiagnosticAudit(
            mccrary=mccrary_mock,
            mccrary_status=mccrary_status, mccrary_msg="ok", mccrary_mentioned=False,
            balance=balance_mock,
            balance_status=balance_status, balance_msg="ok", balance_mentioned=False,
            has_covariates=True,
            placebo=placebo_mock,
            placebo_status=placebo_status, placebo_msg="ok", placebo_mentioned=False,
        )
        return est, bw, frag, diag

    def test_all_pass_gives_robust(self):
        verdict, reasons, positives = _determine_verdict(*self._make_audits())
        assert verdict == "Robust"
        assert reasons == []
        assert len(positives) >= 3

    def test_not_reproduced_gives_problematic(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(reproduced=False, diff_in_se=3.0)
        )
        assert verdict == "Problematic"
        assert any("not reproduced" in r for r in reasons)

    def test_mccrary_fail_gives_problematic(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(mccrary_status="fail")
        )
        assert verdict == "Problematic"

    def test_balance_fail_gives_problematic(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(balance_status="fail")
        )
        assert verdict == "Problematic"

    def test_outside_stable_region_gives_fragile(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(in_stable=False)
        )
        assert verdict == "Fragile"
        assert any("stable region" in r for r in reasons)

    def test_high_bw_ratio_gives_fragile(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(bw_ratio=3.5, in_stable=True)
        )
        assert verdict == "Fragile"
        assert any("over-smoothed" in r for r in reasons)

    def test_low_bw_ratio_gives_fragile(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(bw_ratio=0.25, in_stable=True)
        )
        assert verdict == "Fragile"
        assert any("under-smoothed" in r for r in reasons)

    def test_fragile_frag_status_gives_fragile(self):
        verdict, reasons, _ = _determine_verdict(
            *self._make_audits(frag_status="fragile", frag_score=0.3)
        )
        assert verdict == "Fragile"

    def test_mccrary_warn_gives_fragile(self):
        verdict, _, _ = _determine_verdict(
            *self._make_audits(mccrary_status="warn")
        )
        assert verdict == "Fragile"

    def test_placebo_warn_gives_fragile(self):
        verdict, _, _ = _determine_verdict(
            *self._make_audits(placebo_status="warn")
        )
        assert verdict == "Fragile"


class TestEndToEndVerdicts:
    """Full pipeline: all three pre-loaded examples get the right verdicts."""

    @pytest.mark.parametrize("name,expected", [
        ("Lee 2008 (Electoral)", "Robust"),
        ("Synthetic: Manipulated Design", "Problematic"),
        ("Synthetic: Fragile Design", "Fragile"),
    ])
    def test_preloaded_verdict(self, name, expected):
        examples = get_preloaded_examples()
        ex = examples[name]
        y = ex.df[ex.y_col].values
        x = ex.df[ex.x_col].values
        cov_data = ex.df[ex.cov_cols].values if ex.cov_cols else None
        audit = run_replication_audit(
            y, x, ex.cov_cols, cov_data,
            ex.cutoff, ex.reported_bandwidth,
            ex.reported_estimate, ex.reported_se,
        )
        assert audit.verdict == expected, (
            f"{name}: got '{audit.verdict}', expected '{expected}'\n"
            f"  reasons: {audit.verdict_reasons}"
        )
