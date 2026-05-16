"""Placebo cutoff distribution plot."""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from rd_credibility.diagnostics.placebo_cutoffs import PlaceboResult

_INSIG_COLOR = "#4682B4"
_SIG_COLOR = "#D62728"
_TRUE_COLOR = "#2CA02C"


def _significance_mask(result: PlaceboResult):
    """Boolean mask of which placebos are significant at 5%."""
    est = result.placebo_estimates
    se = result.placebo_ses
    valid = ~(np.isnan(est) | np.isnan(se) | (se <= 0))
    z = np.where(valid, np.abs(est) / se, 0)
    return valid & (z > 1.96)


def plot_interactive(placebo_result: PlaceboResult):
    """
    Interactive placebo estimate distribution plot (Plotly).

    Parameters
    ----------
    placebo_result : PlaceboResult

    Returns
    -------
    plotly.graph_objects.Figure
    """
    r = placebo_result
    estimates = r.placebo_estimates
    sig_mask = _significance_mask(r)

    valid = ~np.isnan(estimates)
    valid_ests = estimates[valid]

    fig = go.Figure()

    if len(valid_ests) >= 3:
        # KDE overlay
        kde_x = np.linspace(valid_ests.min() - 0.5, valid_ests.max() + 0.5, 300)
        kde_y = stats.gaussian_kde(valid_ests)(kde_x)
        fig.add_trace(
            go.Scatter(x=kde_x, y=kde_y, mode="lines",
                       line=dict(color=_INSIG_COLOR, width=2),
                       name="Placebo density", fill="tozeroy",
                       fillcolor=f"rgba(70,130,180,0.15)")
        )

    # Scatter of placebo estimates (rug-style at y=0)
    for i, (est, se, is_sig) in enumerate(zip(r.placebo_estimates, r.placebo_ses, sig_mask)):
        if np.isnan(est):
            continue
        color = _SIG_COLOR if is_sig else _INSIG_COLOR
        fig.add_trace(
            go.Scatter(
                x=[est], y=[0],
                mode="markers",
                marker=dict(color=color, size=8, symbol="line-ns",
                            line=dict(width=2, color=color)),
                showlegend=False,
                hovertemplate=f"Estimate: {est:.3f}<br>SE: {se:.3f}<extra></extra>",
            )
        )

    # True estimate
    fig.add_vline(
        x=r.true_estimate,
        line_color=_TRUE_COLOR,
        line_width=2.5,
        annotation_text=f"True: {r.true_estimate:.3f}",
        annotation_position="top right",
        annotation_font=dict(color=_TRUE_COLOR, size=11),
    )

    n_valid = int(valid.sum())
    n_sig = int(sig_mask.sum())
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.03, y=0.96,
        text=f"<b>{n_sig} of {n_valid} placebo cutoffs significant at 5%</b>",
        showarrow=False, align="left",
        bgcolor="white", bordercolor="gray", borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        title="Placebo Cutoff Test",
        xaxis_title="Placebo RD Estimate",
        yaxis_title="Density",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=True,
    )
    return fig


def plot_publication(placebo_result: PlaceboResult):
    """
    AER-style placebo distribution plot (Matplotlib).

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = placebo_result
    estimates = r.placebo_estimates
    sig_mask = _significance_mask(r)

    valid = ~np.isnan(estimates)
    valid_ests = estimates[valid]
    valid_sig = sig_mask[valid]

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    if len(valid_ests) >= 3:
        ax.hist(valid_ests, bins=max(5, len(valid_ests) // 3),
                color=_INSIG_COLOR, alpha=0.5, edgecolor="white", label="Placebo estimates")

        # Rug marks
        for est, is_sig in zip(valid_ests, valid_sig):
            color = _SIG_COLOR if is_sig else _INSIG_COLOR
            ax.axvline(est, ymin=0, ymax=0.05, color=color, linewidth=1.5)
    else:
        ax.text(0.5, 0.5, "Insufficient placebos", ha="center", va="center",
                transform=ax.transAxes)

    ax.axvline(r.true_estimate, color=_TRUE_COLOR, linewidth=2.0,
               linestyle="-", label=f"True estimate ({r.true_estimate:.3f})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    n_valid = int(valid.sum())
    n_sig = int(sig_mask.sum())
    ax.text(0.03, 0.97,
            f"{n_sig} of {n_valid} placebo cutoffs\nsignificant at 5%",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    handles = [
        mpatches.Patch(color=_INSIG_COLOR, alpha=0.6, label="Insignificant placebo"),
        mpatches.Patch(color=_SIG_COLOR, label="Significant placebo"),
        plt.Line2D([0], [0], color=_TRUE_COLOR, linewidth=2, label="True estimate"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("Placebo RD Estimate", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Placebo Cutoff Test", fontsize=11)
    fig.tight_layout()
    return fig
