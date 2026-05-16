"""McCrary density continuity test page."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Density Test · RD Dashboard", layout="wide")

from rd_credibility.app.components.style import inject_css, page_desc, badge
from rd_credibility.app.components.learning_mode import render_explainer
from rd_credibility.visualization import density_plot

inject_css()

if "results" not in st.session_state:
    st.warning("Return to the main page to load data first.")
    st.stop()

results = st.session_state["results"]
cfg = st.session_state["cfg"]
mccrary = results["mccrary"]

st.title("Density Continuity Test (McCrary)")
page_desc(
    "Tests whether the density of the running variable is smooth at the cutoff — "
    "a violation indicates that units may be sorting into treatment."
)

# ── Status badge ──────────────────────────────────────────────────────────────
p = mccrary.p_value
if p > 0.10:
    status, status_text = "pass", f"PASS — p = {p:.3f}"
elif p > 0.05:
    status, status_text = "warn", f"MARGINAL — p = {p:.3f}"
else:
    status, status_text = "fail", f"FAIL — p = {p:.3f}"

st.markdown(badge(status, status_text), unsafe_allow_html=True)
st.markdown(f"**Conclusion:** {mccrary.conclusion}")

st.markdown("---")

# ── Plot ──────────────────────────────────────────────────────────────────────
st.subheader("Running Variable Density")
fig = density_plot.plot_interactive(mccrary, cutoff=cfg["cutoff"])
st.plotly_chart(fig, use_container_width=True)

# ── Test statistics table ─────────────────────────────────────────────────────
st.subheader("Test Statistics")

stats = {
    "Statistic": [
        "Log density ratio (θ)",
        "Test statistic (z)",
        "p-value",
        "Bins (left)",
        "Bins (right)",
    ],
    "Value": [
        f"{mccrary.theta:.4f}",
        f"{mccrary.t_stat:.3f}",
        f"{p:.4f}",
        str(int((mccrary.bin_centers < cfg["cutoff"]).sum())),
        str(int((mccrary.bin_centers >= cfg["cutoff"]).sum())),
    ],
}

df_stats = pd.DataFrame(stats)
st.dataframe(df_stats, hide_index=True, use_container_width=False)

# ── Learning mode ─────────────────────────────────────────────────────────────
if cfg.get("learning_mode"):
    render_explainer("mccrary", mccrary)
