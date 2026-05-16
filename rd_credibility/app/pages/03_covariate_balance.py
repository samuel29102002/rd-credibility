"""Covariate balance test page."""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Covariate Balance · RD Dashboard", layout="wide")

from rd_credibility.app.components.style import inject_css, page_desc, badge
from rd_credibility.app.components.learning_mode import render_explainer
from rd_credibility.visualization import covariate_grid

inject_css()

if "results" not in st.session_state:
    st.warning("Return to the main page to load data first.")
    st.stop()

results = st.session_state["results"]
cfg = st.session_state["cfg"]
balance = results["balance"]

st.title("Covariate Balance")
page_desc(
    "Runs a separate RD regression for each pre-treatment covariate — "
    "significant discontinuities indicate that units differ systematically across the cutoff."
)

# ── Check for placeholder (no real covariates) ────────────────────────────────
has_covariates = bool(cfg.get("cov_cols"))
if not has_covariates:
    st.info(
        "No pre-treatment covariates were provided for this dataset. "
        "Upload a CSV and map covariate columns to enable this test."
    )
    st.stop()

# ── Status banner ─────────────────────────────────────────────────────────────
n_sig = balance.n_significant
n_total = len(balance.results)
if n_sig == 0:
    st.markdown(badge("pass", f"PASS — 0/{n_total} covariates significant"), unsafe_allow_html=True)
elif n_sig == 1:
    st.markdown(badge("warn", f"WARNING — {n_sig}/{n_total} covariate significant"), unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-banner">⚠️ One covariate shows a significant discontinuity. '
        "This may indicate pre-treatment imbalance at the cutoff.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(badge("fail", f"FAIL — {n_sig}/{n_total} covariates significant"), unsafe_allow_html=True)
    st.markdown(
        f'<div class="error-banner">❌ {n_sig} covariates show significant discontinuities '
        "at the cutoff. This is strong evidence of pre-treatment imbalance.</div>",
        unsafe_allow_html=True,
    )

st.markdown(f"**Overall assessment:** {balance.overall_conclusion}")
st.markdown("---")

# ── Coefficient plot ──────────────────────────────────────────────────────────
st.subheader("Covariate Balance Plot")
st.caption("Estimates should be near zero; red dots indicate significant imbalance (p < 0.05).")
fig = covariate_grid.plot_interactive(balance)
st.plotly_chart(fig, use_container_width=True)

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("Balance Test Results")

df_res = balance.results.copy()
if not df_res.empty:
    df_display = df_res[["covariate", "estimate", "se", "p_value", "significant"]].copy()
    df_display.columns = ["Covariate", "RD Estimate", "Std. Error", "p-value", "Significant"]
    df_display["RD Estimate"] = df_display["RD Estimate"].map(lambda v: f"{v:.4f}")
    df_display["Std. Error"] = df_display["Std. Error"].map(lambda v: f"{v:.4f}")
    df_display["p-value"] = df_display["p-value"].map(lambda v: f"{v:.4f}")
    df_display["Significant"] = df_display["Significant"].map(lambda v: "✓" if v else "")

    st.dataframe(df_display, hide_index=True, use_container_width=True)

# ── Learning mode ─────────────────────────────────────────────────────────────
if cfg.get("learning_mode"):
    render_explainer("balance", balance)
