"""Placebo cutoff test page."""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placebo Tests · RD Dashboard", layout="wide")

from rd_credibility.app.components.style import inject_css, page_desc, badge
from rd_credibility.app.components.learning_mode import render_explainer
from rd_credibility.visualization import placebo_plot

inject_css()

if "results" not in st.session_state:
    st.warning("Return to the main page to load data first.")
    st.stop()

results = st.session_state["results"]
cfg = st.session_state["cfg"]
placebo = results["placebo"]

st.title("Placebo Cutoff Test")
page_desc(
    "Estimates the RD effect at artificial cutoffs where no treatment is assigned — "
    "fewer than 5% of placebo estimates should be statistically significant under a valid design."
)

# ── Status ────────────────────────────────────────────────────────────────────
n_sig = placebo.n_significant_placebos
valid = ~np.isnan(placebo.placebo_estimates)
n_total = int(valid.sum())
rate = n_sig / max(n_total, 1)

if rate <= 0.05:
    st.markdown(badge("pass", f"PASS — {n_sig}/{n_total} significant ({rate:.0%})"), unsafe_allow_html=True)
elif rate <= 0.10:
    st.markdown(badge("warn", f"MARGINAL — {n_sig}/{n_total} significant ({rate:.0%})"), unsafe_allow_html=True)
else:
    st.markdown(badge("fail", f"FAIL — {n_sig}/{n_total} significant ({rate:.0%})"), unsafe_allow_html=True)
    st.markdown(
        f'<div class="error-banner">❌ {n_sig} of {n_total} placebo cutoffs ({rate:.0%}) '
        "are significant at 5%, exceeding the expected chance rate.</div>",
        unsafe_allow_html=True,
    )

st.markdown(f"**True estimate:** {placebo.true_estimate:.4f} (SE = {placebo.true_se:.4f})")
st.markdown(f"**Conclusion:** {placebo.conclusion}")
st.markdown("---")

# ── Distribution plot ─────────────────────────────────────────────────────────
st.subheader("Placebo Estimate Distribution")
st.caption(
    "Blue = insignificant placebo estimates. Red rug marks = significant placebos. "
    "Green vertical line = true RD estimate."
)
fig = placebo_plot.plot_interactive(placebo)
st.plotly_chart(fig, use_container_width=True)

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("Placebo Cutoff Results")
st.caption("Each row is one artificial cutoff. Rows shaded red are significant at 5%.")

ests = placebo.placebo_estimates
ses = placebo.placebo_ses
cutoffs = placebo.placebo_cutoffs

rows = []
for c, e, s in zip(cutoffs, ests, ses):
    if np.isnan(e):
        continue
    z = abs(e) / s if s > 0 else np.nan
    pval = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    rows.append({
        "Placebo cutoff": f"{c:.4f}",
        "Estimate": f"{e:.4f}",
        "SE": f"{s:.4f}",
        "z-stat": f"{z:.2f}" if not np.isnan(z) else "—",
        "p-value": f"{pval:.4f}" if not np.isnan(pval) else "—",
        "Significant": "✓" if (not np.isnan(z) and abs(z) > 1.96) else "",
    })

if rows:
    df_table = pd.DataFrame(rows)

    def _highlight_sig(row):
        if row["Significant"] == "✓":
            return ["background-color: #fdecea"] * len(row)
        return [""] * len(row)

    styled = df_table.style.apply(_highlight_sig, axis=1)
    st.dataframe(styled, hide_index=True, use_container_width=True)

# ── Learning mode ─────────────────────────────────────────────────────────────
if cfg.get("learning_mode"):
    render_explainer("placebo", placebo)
