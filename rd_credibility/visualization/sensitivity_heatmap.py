"""Bandwidth × polynomial order sensitivity heatmap."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from rd_credibility.diagnostics.bandwidth_grid import BandwidthGridResult


def _build_pivot(grid: pd.DataFrame):
    """Pivot grid to (poly_order × bandwidth) arrays for estimate and SE."""
    est_pivot = grid.pivot_table(index="poly_order", columns="bandwidth", values="estimate")
    se_pivot = grid.pivot_table(index="poly_order", columns="bandwidth", values="se")
    return est_pivot, se_pivot


def _baseline_stats(grid: pd.DataFrame, optimal_bw: float):
    """Return (baseline_estimate, baseline_se) at the closest-to-optimal bandwidth, poly=1."""
    order1 = grid[(grid["poly_order"] == 1)].dropna(subset=["estimate", "se"])
    if order1.empty:
        return np.nan, np.nan
    idx = (order1["bandwidth"] - optimal_bw).abs().idxmin()
    row = order1.loc[idx]
    return float(row["estimate"]), float(row["se"])


def plot_interactive(bw_result: BandwidthGridResult):
    """
    Interactive 2-D sensitivity heatmap (Plotly).

    Color = point estimate. Contour lines at baseline ± 1 SE.
    Vertical dashed line at MSE-optimal bandwidth.

    Parameters
    ----------
    bw_result : BandwidthGridResult

    Returns
    -------
    plotly.graph_objects.Figure
    """
    grid = bw_result.grid
    est_pivot, se_pivot = _build_pivot(grid)

    bws = est_pivot.columns.values.astype(float)
    orders = est_pivot.index.values.astype(int)
    z = est_pivot.values.astype(float)

    baseline_est, baseline_se = _baseline_stats(grid, bw_result.optimal_bandwidth)
    contour_vals = [baseline_est - baseline_se, baseline_est + baseline_se]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=bws, y=orders, z=z,
            colorscale="RdYlGn",
            colorbar=dict(title="Estimate"),
            zsmooth="best",
        )
    )

    if not np.isnan(baseline_est):
        for cv in contour_vals:
            fig.add_trace(
                go.Contour(
                    x=bws, y=orders, z=z,
                    contours=dict(
                        start=cv, end=cv, size=0,
                        coloring="none",
                        showlabels=True,
                        labelfont=dict(size=10, color="black"),
                    ),
                    line=dict(color="white", width=1.5, dash="dot"),
                    showscale=False,
                    showlegend=False,
                )
            )

    # MSE-optimal bandwidth line
    fig.add_vline(
        x=bw_result.optimal_bandwidth,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"h*={bw_result.optimal_bandwidth:.3f}",
        annotation_position="top",
        annotation_font=dict(size=11),
    )

    # Stability region rectangle
    sr = bw_result.stable_region
    fig.add_vrect(
        x0=sr[0], x1=sr[1],
        fillcolor="rgba(200,200,255,0.15)",
        layer="below",
        line_width=1.5,
        line_color="blue",
        annotation_text="Stable region",
        annotation_position="top left",
        annotation_font=dict(size=10, color="blue"),
    )

    fig.update_layout(
        title=dict(text="Bandwidth × Polynomial Sensitivity", font=dict(size=16)),
        xaxis_title="Bandwidth",
        yaxis_title="Polynomial order",
        yaxis=dict(tickmode="array", tickvals=list(orders), dtick=1),
        plot_bgcolor="white",
    )
    return fig


def plot_publication(bw_result: BandwidthGridResult):
    """
    AER-style sensitivity heatmap (Matplotlib).

    Returns
    -------
    matplotlib.figure.Figure
    """
    grid = bw_result.grid
    est_pivot, _ = _build_pivot(grid)

    bws = est_pivot.columns.values.astype(float)
    orders = est_pivot.index.values.astype(int)
    z = est_pivot.values.astype(float)

    baseline_est, baseline_se = _baseline_stats(grid, bw_result.optimal_bandwidth)

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Heatmap via pcolormesh with centered ticks
    bw_edges = np.concatenate([[bws[0] - (bws[1] - bws[0]) / 2],
                                (bws[:-1] + bws[1:]) / 2,
                                [bws[-1] + (bws[-1] - bws[-2]) / 2]])
    order_edges = np.arange(orders[0] - 0.5, orders[-1] + 1)

    cmap = plt.get_cmap("RdYlGn")
    mesh = ax.pcolormesh(bw_edges, order_edges, z, cmap=cmap, shading="flat")
    fig.colorbar(mesh, ax=ax, label="Estimate", pad=0.02)

    # Contour lines
    if not np.isnan(baseline_est):
        bw_centers = bws
        order_centers = orders.astype(float)
        BW, OR = np.meshgrid(bw_centers, order_centers)
        cs = ax.contour(BW, OR, z,
                        levels=[baseline_est - baseline_se, baseline_est + baseline_se],
                        colors="white", linewidths=1.2, linestyles="dashed")
        ax.clabel(cs, fmt="%.2f", fontsize=8, colors="white")

    # Optimal bandwidth line
    ax.axvline(bw_result.optimal_bandwidth, color="black",
               linestyle="--", linewidth=1.5,
               label=f"$h^*$ = {bw_result.optimal_bandwidth:.3f}")

    # Stability region
    sr = bw_result.stable_region
    ax.axvspan(sr[0], sr[1], alpha=0.15, color="royalblue")
    ax.annotate("Stable\nregion",
                xy=((sr[0] + sr[1]) / 2, orders[-1] + 0.1),
                ha="center", va="bottom", fontsize=8, color="royalblue")

    ax.set_yticks(orders)
    ax.set_xlabel("Bandwidth", fontsize=10)
    ax.set_ylabel("Polynomial order", fontsize=10)
    ax.set_title("Bandwidth × Polynomial Sensitivity", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig
