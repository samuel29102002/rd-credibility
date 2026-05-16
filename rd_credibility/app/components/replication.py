"""Replication audit engine: reproduce a published RD spec and stress-test it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult, BandwidthSensitivity
from rd_credibility.diagnostics.covariate_balance import CovariateBalance, CovariateBalanceResult
from rd_credibility.diagnostics.mccrary import McCraryResult, McCraryTest
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult, PlaceboTest
from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator
from rd_credibility.scoring.credibility import CredibilityScore


# ── Audit result ──────────────────────────────────────────────────────────────

@dataclass
class EstimateAudit:
    our_estimate: float
    our_se: float
    reported_estimate: float
    reported_se: float
    diff: float            # our - reported
    diff_in_se: float      # |diff| / reported_se
    reproduced: bool       # diff_in_se < 2.0
    status: str            # "reproduced" | "marginal" | "failed"


@dataclass
class BandwidthAudit:
    reported_bw: float
    mse_optimal_bw: float
    ratio: float           # reported / mse_optimal
    in_stable_region: bool
    stable_region: tuple
    status: str            # "optimal" | "acceptable" | "suspicious"


@dataclass
class FragilityAudit:
    fragility_score: float  # fraction of nearby bws within 1.96*med_se of median estimate
    n_nearby: int
    n_stable: int
    status: str             # "stable" | "moderate" | "fragile"


@dataclass
class DiagnosticAudit:
    mccrary: McCraryResult
    mccrary_status: str     # "pass" | "warn" | "fail"
    mccrary_msg: str
    mccrary_mentioned: bool

    balance: Optional[CovariateBalanceResult]
    balance_status: str
    balance_msg: str
    balance_mentioned: bool
    has_covariates: bool

    placebo: PlaceboResult
    placebo_status: str
    placebo_msg: str
    placebo_mentioned: bool

    @property
    def any_warning(self) -> bool:
        return any(s in ("warn", "fail") for s in
                   [self.mccrary_status, self.balance_status, self.placebo_status])


@dataclass
class ReplicationAudit:
    estimate: EstimateAudit
    bandwidth: BandwidthAudit
    fragility: FragilityAudit
    diagnostics: DiagnosticAudit
    bw_grid: BandwidthGridResult
    credibility_score: float
    credibility_grade: str
    verdict: str            # "Robust" | "Fragile" | "Problematic"
    verdict_reasons: list = field(default_factory=list)
    verdict_positives: list = field(default_factory=list)


# ── Pre-loaded examples ───────────────────────────────────────────────────────

@dataclass
class PreloadedExample:
    name: str
    description: str
    paper_ref: str
    expected_verdict: str
    df: pd.DataFrame
    x_col: str
    y_col: str
    cov_cols: list
    cutoff: float
    reported_bandwidth: float
    reported_estimate: float
    reported_se: float


def _build_lee_example() -> PreloadedExample:
    """
    Lee (2008) — clean electoral RD. Should audit as Robust.
    Reported spec matches MSE-optimal; all diagnostics pass.
    """
    from rd_credibility.app.components.data_loader import _lee_2008
    df = _lee_2008(seed=42)
    y, x = df["y"].values, df["x"].values

    bw = mse_optimal_bandwidth(y, x, 0.0)
    rd = RDEstimator(y, x, cutoff=0.0, bandwidth=bw, poly_order=1).fit()

    return PreloadedExample(
        name="Lee 2008 (Electoral)",
        description=(
            "Lee (2008) studies whether winning a close election causes incumbency advantage. "
            "The running variable is the Democratic vote-share margin; the cutoff is 0 (bare majority). "
            "The design is considered a benchmark for clean RD identification."
        ),
        paper_ref="Lee, D.S. (2008). Randomized experiments from non-random selection in U.S. "
                  "House elections. Journal of Econometrics, 142(2), 675–697.",
        expected_verdict="Robust",
        df=df,
        x_col="x", y_col="y",
        cov_cols=["z_lag_vote", "z_turnout", "z_urban"],
        cutoff=0.0,
        reported_bandwidth=round(bw, 4),
        reported_estimate=round(rd.estimate, 4),
        reported_se=round(rd.se, 4),
    )


def _build_manipulated_example() -> PreloadedExample:
    """
    Synthetic manipulated design — should audit as Problematic.
    Strong density discontinuity + estimate not faithfully disclosed.
    """
    from tests.fixtures.synthetic_rd import generate_manipulation_data
    df = generate_manipulation_data(n=3000, manipulation_frac=0.30, seed=77)
    df = df.rename(columns={"x": "x", "y": "y"})
    df["z_income"] = np.random.default_rng(77).normal(50 + 5 * df["x"], 10, len(df))

    y, x = df["y"].values, df["x"].values
    bw_mse = mse_optimal_bandwidth(y, x, 0.0)
    rd_true = RDEstimator(y, x, cutoff=0.0, bandwidth=bw_mse, poly_order=1).fit()

    # Claim the true tau=1.0 even though manipulation inflates the estimate
    reported_estimate = 1.00
    reported_se = 0.09

    return PreloadedExample(
        name="Synthetic: Manipulated Design",
        description=(
            "This synthetic dataset mimics a study where 30% of units just below the cutoff "
            "have sorted themselves above it. The paper claims an effect of τ=1.0 and does not "
            "report the McCrary test. The manipulation creates a sharp density jump that "
            "invalidates the local randomisation assumption."
        ),
        paper_ref="Synthetic example — illustrates the replication crisis pattern where "
                  "manipulation goes unreported.",
        expected_verdict="Problematic",
        df=df,
        x_col="x", y_col="y",
        cov_cols=["z_income"],
        cutoff=0.0,
        reported_bandwidth=round(bw_mse * 1.2, 4),
        reported_estimate=reported_estimate,
        reported_se=reported_se,
    )


def _build_fragile_example() -> PreloadedExample:
    """
    Synthetic fragile design — should audit as Fragile.
    Estimate reproduces but the reported bandwidth is far outside the stable region;
    the outcome function is strongly nonlinear so wider bandwidths introduce heavy bias.
    """
    from tests.fixtures.synthetic_rd import generate_rd_data
    # Strong nonlinearity: opposite slopes create a kink at the cutoff in Y0
    df = generate_rd_data(
        n=2500, tau=0.8, slope_left=2.5, slope_right=-1.0, noise=0.9, seed=13
    )
    df["z_edu"] = np.random.default_rng(13).normal(12 + 0.5 * df["x"], 2, len(df))
    df["z_age"] = np.random.default_rng(14).normal(38 + 1.0 * df["x"], 5, len(df))

    y, x = df["y"].values, df["x"].values
    bw_mse = mse_optimal_bandwidth(y, x, 0.0)

    # The "paper" uses a wide bandwidth — roughly 3x MSE-optimal, outside stable region.
    # Both the paper and our replication code run at this bandwidth, so the estimate
    # will reproduce — but the sensitivity analysis exposes the fragility.
    reported_bw = min(round(bw_mse * 3.2, 4), 0.55)
    rd_wide = RDEstimator(y, x, cutoff=0.0, bandwidth=reported_bw, poly_order=1).fit()

    return PreloadedExample(
        name="Synthetic: Fragile Design",
        description=(
            "This synthetic dataset has a real treatment effect (τ=0.8) but a strongly nonlinear "
            "outcome function. The paper uses a bandwidth ≈3× larger than MSE-optimal. "
            "The estimate reproduces at their exact specification, but the sensitivity analysis "
            "reveals it sits outside the stable region — nearby bandwidths give very different estimates."
        ),
        paper_ref="Synthetic example — illustrates specification fragility without overt fraud.",
        expected_verdict="Fragile",
        df=df,
        x_col="x", y_col="y",
        cov_cols=["z_edu", "z_age"],
        cutoff=0.0,
        reported_bandwidth=reported_bw,
        reported_estimate=round(rd_wide.estimate, 4),
        reported_se=round(rd_wide.se, 4),
    )


def get_preloaded_examples() -> dict[str, PreloadedExample]:
    return {
        "Lee 2008 (Electoral)": _build_lee_example(),
        "Synthetic: Manipulated Design": _build_manipulated_example(),
        "Synthetic: Fragile Design": _build_fragile_example(),
    }


# ── Core audit logic ──────────────────────────────────────────────────────────

def run_replication_audit(
    y: np.ndarray,
    x: np.ndarray,
    cov_cols: list,
    cov_data: Optional[np.ndarray],
    cutoff: float,
    reported_bandwidth: float,
    reported_estimate: float,
    reported_se: float,
    poly_order: int = 1,
    kernel: str = "triangular",
) -> ReplicationAudit:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # 1 ── Reproduce at their reported bandwidth
    rd = RDEstimator(y, x, cutoff=cutoff, bandwidth=reported_bandwidth,
                     poly_order=poly_order, kernel=kernel).fit()

    diff = rd.estimate - reported_estimate
    diff_in_se = abs(diff) / max(reported_se, 1e-12)
    if diff_in_se < 1.0:
        est_status = "reproduced"
    elif diff_in_se < 2.0:
        est_status = "marginal"
    else:
        est_status = "failed"

    estimate_audit = EstimateAudit(
        our_estimate=rd.estimate,
        our_se=rd.se,
        reported_estimate=reported_estimate,
        reported_se=reported_se,
        diff=diff,
        diff_in_se=diff_in_se,
        reproduced=(diff_in_se < 2.0),
        status=est_status,
    )

    # 2 ── MSE-optimal bandwidth
    mse_bw = mse_optimal_bandwidth(y, x, cutoff)
    ratio = reported_bandwidth / max(mse_bw, 1e-9)

    # 3 ── Full bandwidth grid (centred on their spec for tighter resolution)
    x_range = float(x.max() - x.min())
    bw_lo = max(reported_bandwidth * 0.20, x_range * 0.01)
    bw_hi = min(reported_bandwidth * 4.0, x_range * 0.90)
    bw_grid = BandwidthSensitivity(
        y, x, cutoff=cutoff,
        bw_range=(bw_lo, bw_hi),
        poly_orders=[1, 2, 3],
    ).fit()

    sr = bw_grid.stable_region
    bw_in_stable = bool(sr[0] <= reported_bandwidth <= sr[1])

    if ratio < 1.5 and bw_in_stable:
        bw_status = "optimal"
    elif bw_in_stable:
        bw_status = "acceptable"
    else:
        bw_status = "suspicious"

    bw_audit = BandwidthAudit(
        reported_bw=reported_bandwidth,
        mse_optimal_bw=mse_bw,
        ratio=ratio,
        in_stable_region=bw_in_stable,
        stable_region=sr,
        status=bw_status,
    )

    # 4 ── Fragility: fraction of nearby bws (0.5x–2x) giving stable estimates
    grid_p1 = bw_grid.grid[bw_grid.grid["poly_order"] == 1].dropna(subset=["estimate", "se"])
    nearby = grid_p1[
        (grid_p1["bandwidth"] >= reported_bandwidth * 0.40) &
        (grid_p1["bandwidth"] <= reported_bandwidth * 2.50)
    ]
    if len(nearby) >= 2:
        med_est = float(nearby["estimate"].median())
        med_se = float(nearby["se"].median())
        n_stable = int(((nearby["estimate"] - med_est).abs() <= 1.96 * med_se).sum())
        frag_score = n_stable / len(nearby)
    else:
        n_stable = len(nearby)
        frag_score = 1.0

    if frag_score >= 0.75:
        frag_status = "stable"
    elif frag_score >= 0.50:
        frag_status = "moderate"
    else:
        frag_status = "fragile"

    frag_audit = FragilityAudit(
        fragility_score=frag_score,
        n_nearby=len(nearby),
        n_stable=n_stable,
        status=frag_status,
    )

    # 5 ── McCrary test
    mccrary = McCraryTest(x, cutoff=cutoff).fit()
    if mccrary.p_value > 0.10:
        mc_status, mc_msg = "pass", f"No evidence of sorting (p = {mccrary.p_value:.3f})"
    elif mccrary.p_value > 0.05:
        mc_status, mc_msg = "warn", f"Marginal density discontinuity (p = {mccrary.p_value:.3f})"
    else:
        mc_status, mc_msg = "fail", f"Significant manipulation detected (p = {mccrary.p_value:.3f})"

    # 6 ── Covariate balance
    has_covariates = cov_data is not None and len(cov_cols) > 0
    if has_covariates:
        covdf = pd.DataFrame(cov_data, columns=cov_cols)
        balance = CovariateBalance(x, covdf, cutoff=cutoff, bandwidth=reported_bandwidth).fit()
        n_sig = balance.n_significant
        n_total = len(balance.results)
        threshold = max(1, math.ceil(0.10 * n_total))
        if n_sig == 0:
            bal_status, bal_msg = "pass", f"0/{n_total} covariates imbalanced"
        elif n_sig <= threshold:
            bal_status, bal_msg = "warn", f"{n_sig}/{n_total} covariate(s) significant (≤ chance)"
        else:
            bal_status, bal_msg = "fail", f"{n_sig}/{n_total} covariates significantly imbalanced"
    else:
        balance = None
        bal_status, bal_msg = "pass", "No covariates provided"

    # 7 ── Placebo test
    placebo = PlaceboTest(y, x, cutoff=cutoff, bandwidth=reported_bandwidth, n_placebo=20).fit()
    valid_placebos = int((~np.isnan(placebo.placebo_estimates)).sum())
    n_sig_plac = placebo.n_significant_placebos
    plac_rate = n_sig_plac / max(valid_placebos, 1)
    if plac_rate <= 0.05:
        plac_status, plac_msg = "pass", f"{n_sig_plac}/{valid_placebos} placebo cutoffs significant ({plac_rate:.0%})"
    elif plac_rate <= 0.10:
        plac_status, plac_msg = "warn", f"{n_sig_plac}/{valid_placebos} significant ({plac_rate:.0%}) — above 5% threshold"
    else:
        plac_status, plac_msg = "fail", f"{n_sig_plac}/{valid_placebos} significant ({plac_rate:.0%}) — far above expected"

    diag_audit = DiagnosticAudit(
        mccrary=mccrary,
        mccrary_status=mc_status,
        mccrary_msg=mc_msg,
        mccrary_mentioned=False,
        balance=balance,
        balance_status=bal_status,
        balance_msg=bal_msg,
        balance_mentioned=False,
        has_covariates=has_covariates,
        placebo=placebo,
        placebo_status=plac_status,
        placebo_msg=plac_msg,
        placebo_mentioned=False,
    )

    # 8 ── Credibility score
    # Use a placeholder empty-covariate balance if no covariates
    if balance is None:
        balance_for_score = CovariateBalance(
            x, pd.DataFrame({"_none": np.zeros(len(x))}),
            cutoff=cutoff, bandwidth=reported_bandwidth,
        ).fit()
    else:
        balance_for_score = balance

    report = CredibilityScore(
        mccrary_result=mccrary,
        balance_result=balance_for_score,
        bandwidth_result=bw_grid,
        placebo_result=placebo,
    ).compute()

    # 9 ── Verdict
    verdict, reasons, positives = _determine_verdict(
        estimate_audit, bw_audit, frag_audit, diag_audit
    )

    return ReplicationAudit(
        estimate=estimate_audit,
        bandwidth=bw_audit,
        fragility=frag_audit,
        diagnostics=diag_audit,
        bw_grid=bw_grid,
        credibility_score=report.total_score,
        credibility_grade=report.grade,
        verdict=verdict,
        verdict_reasons=reasons,
        verdict_positives=positives,
    )


def _determine_verdict(
    est: EstimateAudit,
    bw: BandwidthAudit,
    frag: FragilityAudit,
    diag: DiagnosticAudit,
) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    positives: list[str] = []

    # ── Problematic criteria (hard fails) ─────────────────────────────────────
    is_problematic = False

    if not est.reproduced:
        is_problematic = True
        reasons.append(
            f"Estimate not reproduced: reported {est.reported_estimate:.3f} but we get "
            f"{est.our_estimate:.3f} ({est.diff_in_se:.1f}× the reported SE)."
        )

    if diag.mccrary_status == "fail":
        is_problematic = True
        reasons.append(f"Density manipulation detected: {diag.mccrary_msg}.")

    if diag.balance_status == "fail":
        is_problematic = True
        reasons.append(f"Severe covariate imbalance: {diag.balance_msg}.")

    if is_problematic:
        return "Problematic", reasons, positives

    # ── Fragile criteria ──────────────────────────────────────────────────────
    is_fragile = False

    if not bw.in_stable_region:
        is_fragile = True
        reasons.append(
            f"Reported bandwidth ({bw.reported_bw:.3f}) is outside the stable region "
            f"[{bw.stable_region[0]:.3f}, {bw.stable_region[1]:.3f}]."
        )

    if bw.ratio > 3.0:
        is_fragile = True
        reasons.append(
            f"Reported bandwidth is {bw.ratio:.1f}× the MSE-optimal — "
            "likely over-smoothed and biased."
        )
    elif bw.ratio < 0.30:
        is_fragile = True
        reasons.append(
            f"Reported bandwidth is only {bw.ratio:.2f}× the MSE-optimal — "
            "likely under-smoothed and noisy."
        )

    if frag.status in ("fragile", "moderate"):
        is_fragile = True
        reasons.append(
            f"Specification is {frag.status}: only {frag.n_stable}/{frag.n_nearby} "
            f"nearby bandwidths give consistent estimates ({frag.fragility_score:.0%})."
        )

    if diag.mccrary_status == "warn":
        is_fragile = True
        reasons.append(f"Marginal manipulation signal: {diag.mccrary_msg}.")

    if diag.balance_status == "warn":
        is_fragile = True
        reasons.append(f"Borderline covariate imbalance: {diag.balance_msg}.")

    if diag.placebo_status in ("warn", "fail"):
        is_fragile = True
        reasons.append(f"Elevated placebo rate: {diag.placebo_msg}.")

    # Collect positives for the Robust / Fragile verdict display
    if est.reproduced:
        positives.append(
            f"Estimate reproduced within {est.diff_in_se:.1f} reported SEs "
            f"({est.our_estimate:.3f} vs {est.reported_estimate:.3f})."
        )
    if bw.in_stable_region:
        positives.append(
            f"Bandwidth is within the stable region "
            f"[{bw.stable_region[0]:.3f}, {bw.stable_region[1]:.3f}]."
        )
    if frag.status == "stable":
        positives.append(
            f"Specification is stable: {frag.n_stable}/{frag.n_nearby} "
            f"nearby bandwidths agree ({frag.fragility_score:.0%})."
        )
    if diag.mccrary_status == "pass":
        positives.append(f"No manipulation detected ({diag.mccrary_msg}).")
    if diag.balance_status == "pass" and diag.has_covariates:
        positives.append(f"Covariate balance confirmed ({diag.balance_msg}).")
    if diag.placebo_status == "pass":
        positives.append(f"Placebo tests pass ({diag.placebo_msg}).")

    if is_fragile:
        return "Fragile", reasons, positives

    return "Robust", reasons, positives
