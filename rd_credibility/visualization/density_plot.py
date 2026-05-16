"""McCrary density continuity plot."""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from rd_credibility.diagnostics.mccrary import McCraryResult

_LEFT_COLOR = "#4682B4"
_RIGHT_COLOR = "#E8735A"


def plot_interactive(mccrary_result: McCraryResult, cutoff: float = 0.0):
    """
    Interactive density continuity plot (Plotly).

    Parameters
    ----------
    mccrary_result : McCraryResult
    cutoff : float

    Returns
    -------
    plotly.graph_objects.Figure
    """
    r = mccrary_result
    bc = r.bin_centers
    left_mask = bc < cutoff
    right_mask = bc >= cutoff

    s = bc[1] - bc[0] if len(bc) > 1 else 0.05
    bin_densities = r.bin_counts / (r.bin_counts.sum() * s) if r.bin_counts.sum() > 0 else r.bin_counts

    fig = go.Figure()

    # Histogram bars
    for mask, color, name in [
        (left_mask, _LEFT_COLOR, "Left side"),
        (right_mask, _RIGHT_COLOR, "Right side"),
    ]:
        fig.add_trace(
            go.Bar(
                x=bc[mask], y=bin_densities[mask],
                width=s * 0.9,
                marker_color=color,
                marker_opacity=0.5,
                name=name,
            )
        )

    # Fitted curves
    if len(r.fitted_left) > 0:
        fig.add_trace(
            go.Scatter(
                x=bc[left_mask], y=r.fitted_left,
                mode="lines",
                line=dict(color=_LEFT_COLOR, width=2.5),
                showlegend=False,
            )
        )
    if len(r.fitted_right) > 0:
        fig.add_trace(
            go.Scatter(
                x=bc[right_mask], y=r.fitted_right,
                mode="lines",
                line=dict(color=_RIGHT_COLOR, width=2.5),
                showlegend=False,
            )
        )

    fig.add_vline(
        x=cutoff, line_dash="dash", line_color="black", line_width=1.5,
        annotation_text="Cutoff", annotation_position="top right",
    )

    p_str = f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "< 0.0001"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.95,
        text=f"<b>{r.conclusion}</b><br>p-value: {p_str}",
        showarrow=False, align="left",
        bgcolor="white", bordercolor="gray", borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        title="Running Variable Density (McCrary Test)",
        xaxis_title="Running variable",
        yaxis_title="Density",
        barmode="overlay",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def plot_publication(mccrary_result: McCraryResult, cutoff: float = 0.0):
    """
    AER-style density continuity plot (Matplotlib).

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = mccrary_result
    bc = r.bin_centers
    left_mask = bc < cutoff
    right_mask = bc >= cutoff

    s = bc[1] - bc[0] if len(bc) > 1 else 0.05
    bin_densities = r.bin_counts / (r.bin_counts.sum() * s) if r.bin_counts.sum() > 0 else r.bin_counts

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.bar(bc[left_mask], bin_densities[left_mask], width=s * 0.9,
           color=_LEFT_COLOR, alpha=0.45, label="Left side")
    ax.bar(bc[right_mask], bin_densities[right_mask], width=s * 0.9,
           color=_RIGHT_COLOR, alpha=0.45, label="Right side")

    if len(r.fitted_left) > 0:
        ax.plot(bc[left_mask], r.fitted_left, color=_LEFT_COLOR, linewidth=2.0)
    if len(r.fitted_right) > 0:
        ax.plot(bc[right_mask], r.fitted_right, color=_RIGHT_COLOR, linewidth=2.0)

    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2)

    p_str = f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "< 0.0001"
    ax.text(0.03, 0.94,
            f"{r.conclusion}\np = {p_str}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("Running variable", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Running Variable Density (McCrary Test)", fontsize=11)
    fig.tight_layout()
    return fig
