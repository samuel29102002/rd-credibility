"""Overview page: RD plot, KPI cards, and credibility score gauge."""

import numpy as np
import streamlit as st

st.set_page_config(page_title="Overview · RD Dashboard", layout="wide")

from rd_credibility.app.components.style import inject_css, page_desc, kpi_card
from rd_credibility.app.components.learning_mode import render_explainer
from rd_credibility.visualization import rd_plot, score_gauge

inject_css()

if "results" not in st.session_state:
    st.warning("Return to the main page to load data first.")
    st.stop()

results = st.session_state["results"]
cfg = st.session_state["cfg"]
y = st.session_state["y"]
x = st.session_state["x"]

rd_res = results["rd_result"]
bw = results["bandwidth_used"]
report = results["report"]

st.title("Overview")
page_desc(
    "The RD scatter plot with fitted local polynomial, key estimation statistics, "
    "and the composite RD Credibility Score."
)

# ── RD plot ───────────────────────────────────────────────────────────────────
st.subheader("Regression Discontinuity Plot")
fig = rd_plot.plot_interactive(
    y, x,
    cutoff=cfg["cutoff"],
    bandwidth=bw,
    poly_order=cfg["poly_order"],
)
st.plotly_chart(fig, use_container_width=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.subheader("Key Statistics")
in_bw = np.abs(x - cfg["cutoff"]) <= bw
n_in_bw = int(in_bw.sum())
ci_str = f"[{rd_res.ci_lower:.3f}, {rd_res.ci_upper:.3f}]"

cols = st.columns(4)
cards = [
    ("RD Estimate", f"{rd_res.estimate:.3f}"),
    ("Std. Error", f"{rd_res.se:.3f}"),
    ("Optimal BW", f"{bw:.3f}"),
    ("N in BW", f"{n_in_bw:,}"),
]
for col, (label, value) in zip(cols, cards):
    col.markdown(kpi_card(label, value), unsafe_allow_html=True)

st.caption(
    f"95% CI: {ci_str} · p-value: {rd_res.p_value:.4f} · "
    f"Kernel: {cfg['kernel']} · Poly order: {cfg['poly_order']}"
)

st.markdown("---")

# ── Credibility score ─────────────────────────────────────────────────────────
st.subheader("RD Credibility Score")
page_desc(
    "The composite score aggregates four diagnostic dimensions — each scored out of 25 — "
    "into a single credibility grade."
)

fig_gauge = score_gauge.plot_interactive(report)
st.plotly_chart(fig_gauge, use_container_width=True)

if report.warnings:
    for w in report.warnings:
        st.markdown(f'<div class="error-banner">⚠️ {w}</div>', unsafe_allow_html=True)

if cfg.get("learning_mode"):
    render_explainer("credibility_score", report)
