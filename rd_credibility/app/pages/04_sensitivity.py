"""Bandwidth and polynomial sensitivity analysis page."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sensitivity · RD Dashboard", layout="wide")

from rd_credibility.app.components.style import inject_css, page_desc
from rd_credibility.app.components.learning_mode import render_explainer
from rd_credibility.visualization import sensitivity_heatmap

inject_css()

if "results" not in st.session_state:
    st.warning("Return to the main page to load data first.")
    st.stop()

results = st.session_state["results"]
cfg = st.session_state["cfg"]
bw_res = results["bw_grid"]

st.title("Bandwidth Sensitivity")
page_desc(
    "Shows how the RD estimate changes across different bandwidths and polynomial orders — "
    "a robust design produces stable estimates in the stable region."
)

sr = bw_res.stable_region
opt_bw = bw_res.optimal_bandwidth
st.markdown(
    f"**MSE-optimal bandwidth:** {opt_bw:.3f} · "
    f"**Stable region:** [{sr[0]:.3f}, {sr[1]:.3f}]"
)
st.markdown("---")

# ── Heatmap (visual centrepiece) ──────────────────────────────────────────────
st.subheader("Sensitivity Heatmap")
st.caption(
    "Color = RD estimate. White contour lines mark ±1 SE from the baseline (poly=1, optimal BW). "
    "Shaded band = stable region. Dashed line = MSE-optimal bandwidth."
)
fig_heat = sensitivity_heatmap.plot_interactive(bw_res)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Line chart: estimate vs bandwidth per poly order ──────────────────────────
st.subheader("Estimate vs. Bandwidth by Polynomial Order")
st.caption("Each line traces one polynomial order; the shaded stable region is shown in blue.")

grid = bw_res.grid.dropna(subset=["estimate", "se"])
colors = {1: "#1a1a2e", 2: "#2980b9", 3: "#8e44ad"}
dash_styles = {1: "solid", 2: "dash", 3: "dot"}

fig_line = go.Figure()

# Stable region fill
fig_line.add_vrect(
    x0=sr[0], x1=sr[1],
    fillcolor="rgba(41,128,185,0.10)",
    layer="below",
    line_width=1.2,
    line_color="rgba(41,128,185,0.6)",
)

for order in sorted(grid["poly_order"].unique()):
    sub = grid[grid["poly_order"] == order].sort_values("bandwidth")
    c = colors.get(order, "#555")
    ds = dash_styles.get(order, "solid")

    # CI band
    fig_line.add_trace(go.Scatter(
        x=np.concatenate([sub["bandwidth"], sub["bandwidth"][::-1]]),
        y=np.concatenate([
            sub["estimate"] + 1.96 * sub["se"],
            (sub["estimate"] - 1.96 * sub["se"])[::-1],
        ]),
        fill="toself",
        fillcolor={1: "rgba(26,26,46,0.08)", 2: "rgba(41,128,185,0.08)", 3: "rgba(142,68,173,0.08)"}.get(order, "rgba(85,85,85,0.08)"),
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_line.add_trace(go.Scatter(
        x=sub["bandwidth"], y=sub["estimate"],
        mode="lines+markers",
        name=f"Poly {order}",
        line=dict(color=c, width=2, dash=ds),
        marker=dict(size=4),
    ))

fig_line.add_vline(
    x=opt_bw, line_dash="dash", line_color="#c0392b", line_width=1.5,
    annotation_text=f"h* = {opt_bw:.3f}",
    annotation_font=dict(color="#c0392b", size=11),
)

fig_line.update_layout(
    xaxis_title="Bandwidth",
    yaxis_title="RD Estimate",
    plot_bgcolor="white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=400,
)
st.plotly_chart(fig_line, use_container_width=True)

# ── Learning mode ─────────────────────────────────────────────────────────────
if cfg.get("learning_mode"):
    render_explainer("sensitivity", bw_res)
