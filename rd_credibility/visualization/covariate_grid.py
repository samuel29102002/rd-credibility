"""Covariate balance coefficient plot."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rd_credibility.diagnostics.covariate_balance import CovariateBalanceResult

_GREEN = "#2CA02C"
_ORANGE = "#FF7F0E"
_RED = "#D62728"


def _color_from_pvalue(p):
    if p < 0.05:
        return _RED
    elif p < 0.10:
        return _ORANGE
    return _GREEN


def _sorted_df(result: CovariateBalanceResult):
    """Return result.results sorted by p_value ascending (significant first)."""
    df = result.results.dropna(subset=["estimate", "se", "p_value"]).copy()
    df["ci_lower"] = df["estimate"] - 1.96 * df["se"]
    df["ci_upper"] = df["estimate"] + 1.96 * df["se"]
    df["color"] = df["p_value"].apply(_color_from_pvalue)
    return df.sort_values("p_value")


def plot_interactive(balance_result: CovariateBalanceResult):
    """
    Interactive covariate balance coefficient plot (Plotly).

    Parameters
    ----------
    balance_result : CovariateBalanceResult

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = _sorted_df(balance_result)
    if df.empty:
        return go.Figure().update_layout(title="No covariate data available")

    fig = go.Figure()

    for _, row in df.iterrows():
        color = _color_from_pvalue(row["p_value"])
        fig.add_trace(
            go.Scatter(
                x=[row["ci_lower"], row["ci_upper"]],
                y=[row["covariate"], row["covariate"]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["estimate"]],
                y=[row["covariate"]],
                mode="markers",
                marker=dict(color=color, size=10),
                name=row["covariate"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['covariate']}</b><br>"
                    f"Estimate: {row['estimate']:.3f}<br>"
                    f"SE: {row['se']:.3f}<br>"
                    f"p: {row['p_value']:.3f}<extra></extra>"
                ),
            )
        )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1.0)

    # Legend annotation
    legend_text = (
        "● p < 0.05 (imbalanced)  "
        "● p < 0.10 (marginal)  "
        "● p ≥ 0.10 (balanced)"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        text=legend_text,
        showarrow=False,
        font=dict(size=10),
    )

    fig.update_layout(
        title="Covariate Balance at RD Cutoff",
        xaxis_title="RD Estimate (should be ≈ 0)",
        yaxis_title="",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(b=60),
    )
    return fig


def plot_publication(balance_result: CovariateBalanceResult):
    """
    AER-style covariate balance dot plot (Matplotlib).

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = _sorted_df(balance_result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No covariate data", ha="center", va="center")
        return fig

    n = len(df)
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.5 * n + 1.2)))

    yticks = list(range(n))
    names = df["covariate"].tolist()

    for i, (_, row) in enumerate(df.iterrows()):
        color = _color_from_pvalue(row["p_value"])
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], color=color, linewidth=1.8)
        ax.scatter(row["estimate"], i, color=color, s=50, zorder=5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)

    ax.set_yticks(yticks)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("RD Estimate (should be ≈ 0)", fontsize=10)
    ax.set_title("Covariate Balance at RD Cutoff", fontsize=11)

    handles = [
        mpatches.Patch(color=_RED, label="p < 0.05"),
        mpatches.Patch(color=_ORANGE, label="p < 0.10"),
        mpatches.Patch(color=_GREEN, label="p ≥ 0.10"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig
