"""RD Credibility Dashboard — entry point."""

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="RD Credibility Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "RD Credibility Dashboard — regression discontinuity diagnostic suite."},
)

from rd_credibility.app.components.sidebar import render_sidebar
from rd_credibility.app.components.runner import run_all
from rd_credibility.app.components.style import inject_css, page_desc, kpi_card
from rd_credibility.visualization import rd_plot, score_gauge

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
cfg = render_sidebar()

if cfg["df"] is None:
    st.title("RD Credibility Dashboard")
    st.info("Select a dataset from the sidebar to begin.")
    st.stop()

df = cfg["df"]
y = df[cfg["y_col"]].values
x = df[cfg["x_col"]].values
cov_cols = cfg["cov_cols"]
cov_data = df[cov_cols].values if cov_cols else None

# ── Run diagnostics (cached) ──────────────────────────────────────────────────
with st.spinner("Running diagnostics…"):
    results = run_all(
        y_vals=y,
        x_vals=x,
        cov_cols=cov_cols,
        cov_data=cov_data,
        cutoff=cfg["cutoff"],
        bandwidth=cfg["bandwidth"],
        poly_order=cfg["poly_order"],
        kernel=cfg["kernel"],
    )

# Store in session state for pages
st.session_state["results"] = results
st.session_state["cfg"] = cfg
st.session_state["y"] = y
st.session_state["x"] = x

# ── Overview page ─────────────────────────────────────────────────────────────
st.title("RD Credibility Dashboard")
page_desc(
    f"Overview for <b>{cfg['dataset_name']}</b> — RD scatter plot, "
    "key statistics, and the composite credibility score."
)

rd_res = results["rd_result"]
bw = results["bandwidth_used"]
report = results["report"]

# RD plot
st.subheader("Regression Discontinuity Plot")
fig_rd = rd_plot.plot_interactive(
    y, x,
    cutoff=cfg["cutoff"],
    bandwidth=bw,
    poly_order=cfg["poly_order"],
)
st.plotly_chart(fig_rd, use_container_width=True)

# KPI cards
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

st.caption(f"95% CI: {ci_str} · p-value: {rd_res.p_value:.4f} · Poly order: {cfg['poly_order']}")

st.markdown("---")

# Credibility score
st.subheader("Credibility Score")
fig_gauge = score_gauge.plot_interactive(report)
st.plotly_chart(fig_gauge, use_container_width=True)

# Warnings
if report.warnings:
    for w in report.warnings:
        st.markdown(f'<div class="error-banner">⚠️ {w}</div>', unsafe_allow_html=True)

st.caption(
    "Navigate the sidebar pages to explore individual diagnostic tests in detail."
)
