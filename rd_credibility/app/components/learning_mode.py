"""Learning mode: styled explainer boxes for each diagnostic."""

import streamlit as st


_EXPLAINERS = {
    "mccrary": {
        "title": "McCrary Density Test",
        "what": (
            "Tests whether the density of the running variable is continuous at the cutoff. "
            "A sharp jump in density suggests that units may be manipulating their assignment — "
            "for example, students barely missing a scholarship threshold retaking exams until they pass."
        ),
        "why": (
            "RD validity rests on local randomisation near the cutoff: units just below and just "
            "above should be similar on all pre-treatment characteristics. Sorting invalidates this "
            "assumption because treated and control units near the cutoff are no longer comparable."
        ),
        "threshold": (
            "**p-value > 0.10**: No evidence of manipulation — full score (25/25). "
            "**0.05 < p ≤ 0.10**: Marginal evidence — partial score (15/25). "
            "**p ≤ 0.05**: Significant manipulation detected — zero score, and if p < 0.01 "
            "a hard ceiling of 25/100 is imposed on the total credibility score."
        ),
    },
    "balance": {
        "title": "Covariate Balance",
        "what": (
            "Runs a separate RD regression for each pre-treatment covariate as the outcome. "
            "Under a valid design, no covariate should exhibit a discontinuity at the cutoff — "
            "there is no 'treatment effect' on a variable measured before treatment was assigned."
        ),
        "why": (
            "Even if the density looks smooth, units may still differ systematically across "
            "the cutoff on observed characteristics. Balance tests provide direct evidence "
            "that the quasi-experimental assumption holds for observable confounders."
        ),
        "threshold": (
            "The test flags imbalance when more covariates are significant at 5% than we would "
            "expect by chance (threshold: max(1, ceil(10% × n_covariates))). "
            "All pass → 25/25. Any significant → score penalised proportionally."
        ),
    },
    "sensitivity": {
        "title": "Bandwidth Sensitivity",
        "what": (
            "Estimates the RD effect across a grid of bandwidths (from narrow to wide) and "
            "polynomial orders (1–3). A credible estimate should not vary dramatically with "
            "reasonable changes to these specification choices."
        ),
        "why": (
            "The bandwidth is a researcher degree of freedom: if the estimated effect changes "
            "sign or becomes insignificant for nearby bandwidths, readers cannot trust the "
            "primary specification. Stability signals the discontinuity is real, not an artefact."
        ),
        "threshold": (
            "**Stable region**: bandwidths where the poly-1 estimate stays within 1.96 × median(SE) "
            "of the median estimate. Score is proportional to the fraction of bandwidths in the "
            "stable region, with a bonus when the MSE-optimal bandwidth falls inside it."
        ),
    },
    "placebo": {
        "title": "Placebo Cutoff Test",
        "what": (
            "Estimates the RD effect at a grid of artificial cutoffs where no treatment is "
            "assigned. If the discontinuity at the true cutoff is real, these placebo estimates "
            "should be centred on zero and mostly statistically insignificant."
        ),
        "why": (
            "If we find a big 'effect' at many arbitrary points in the data, the discontinuity "
            "we observe at the true cutoff may simply reflect a noisy outcome function rather "
            "than a genuine treatment effect."
        ),
        "threshold": (
            "Expects ≤ 5% of placebo estimates to be significant at 5% (the type-I error rate "
            "under the null). More than 10% significant → zero score. "
            "Fewer than 5% → full score (25/25)."
        ),
    },
    "credibility_score": {
        "title": "RD Credibility Score",
        "what": (
            "A composite score (0–100) aggregating the four diagnostic dimensions: "
            "manipulation (McCrary), covariate balance, bandwidth sensitivity, and placebo tests. "
            "Each component contributes equally (25 points), and the total maps to a letter grade."
        ),
        "why": (
            "No single diagnostic is sufficient — a design can pass the density test but fail "
            "on balance, or show stable estimates but have significant placebos. "
            "The composite score forces researchers to attend to all four dimensions simultaneously."
        ),
        "threshold": (
            "**A (85–100)**: Publication-ready. "
            "**B (70–84)**: Credible with minor caveats. "
            "**C (55–69)**: Notable weakness — use with caution. "
            "**D (40–54)**: Serious concern. "
            "**F (<40)**: Fundamental validity threat."
        ),
    },
}


def render_explainer(test_name: str, result=None) -> None:
    """
    Render a styled learning-mode explainer box.

    Parameters
    ----------
    test_name : str
        One of: 'mccrary', 'balance', 'sensitivity', 'placebo', 'credibility_score'
    result : optional
        The diagnostic result object; used to generate a one-sentence verdict.
    """
    if test_name not in _EXPLAINERS:
        return

    info = _EXPLAINERS[test_name]
    verdict = _make_verdict(test_name, result)

    with st.expander(f"📖 Learning Mode: {info['title']}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**What this test checks**")
            st.markdown(info["what"])
            st.markdown("**Why it matters**")
            st.markdown(info["why"])
        with col2:
            st.markdown("**Scoring thresholds**")
            st.markdown(info["threshold"])
            if verdict:
                st.markdown("---")
                st.markdown(f"**Verdict on this result**")
                st.markdown(verdict)


def _make_verdict(test_name: str, result) -> str:
    if result is None:
        return ""

    try:
        if test_name == "mccrary":
            p = result.p_value
            if p > 0.10:
                return f"✅ p = {p:.3f} — no evidence of sorting; density continuity assumption satisfied."
            elif p > 0.05:
                return f"⚠️ p = {p:.3f} — marginal evidence of sorting; interpret with caution."
            else:
                return f"❌ p = {p:.3f} — statistically significant density jump; manipulation likely."

        elif test_name == "balance":
            n_sig = result.n_significant
            total = len(result.results)
            if n_sig == 0:
                return f"✅ 0/{total} covariates significant — no imbalance detected."
            else:
                sig_names = result.results[result.results["significant"]]["covariate"].tolist()
                names_str = ", ".join(sig_names[:3])
                return f"⚠️ {n_sig}/{total} covariate(s) significant: {names_str}."

        elif test_name == "sensitivity":
            sr = result.stable_region
            return (
                f"✅ Stable region spans [{sr[0]:.3f}, {sr[1]:.3f}] — "
                f"estimate is robust across this bandwidth range."
                if sr[1] > sr[0] else
                "⚠️ No stable region identified; estimate varies substantially across bandwidths."
            )

        elif test_name == "placebo":
            n_sig = result.n_significant_placebos
            n_total = int((~__import__("numpy").isnan(result.placebo_estimates)).sum())
            rate = n_sig / max(n_total, 1)
            if rate <= 0.05:
                return f"✅ {n_sig}/{n_total} placebo cutoffs significant ({rate:.0%}) — within chance."
            else:
                return f"⚠️ {n_sig}/{n_total} placebo cutoffs significant ({rate:.0%}) — above 5% threshold."

        elif test_name == "credibility_score":
            score = result.total_score
            grade = result.grade
            return f"Overall credibility: **{score:.0f}/100 (Grade {grade})**. {result.summary}"

    except Exception:
        pass
    return ""
