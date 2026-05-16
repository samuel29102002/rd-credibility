"""Credibility score gauge chart and component bars."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rd_credibility.scoring.credibility import CredibilityReport

_GRADE_COLOR = {
    "A": "#2CA02C",
    "B": "#8BC34A",
    "C": "#FFC107",
    "D": "#FF7F0E",
    "F": "#D62728",
}

_ZONE_COLORS = [
    (0, 40, "rgba(214,39,40,0.9)"),
    (40, 55, "rgba(255,127,14,0.9)"),
    (55, 70, "rgba(255,193,7,0.9)"),
    (70, 85, "rgba(139,195,74,0.9)"),
    (85, 100, "rgba(44,160,44,0.9)"),
]

_COMPONENT_LABELS = {
    "manipulation": "Manipulation",
    "balance": "Balance",
    "sensitivity": "Sensitivity",
    "placebo": "Placebo",
}


def _score_to_color(score: float) -> str:
    for lo, hi, color in _ZONE_COLORS:
        if score < hi:
            return color
    return _ZONE_COLORS[-1][2]


def plot_interactive(report: CredibilityReport):
    """
    Interactive credibility gauge + component bar chart (Plotly).

    Parameters
    ----------
    report : CredibilityReport

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "indicator"}], [{"type": "bar"}]],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.04,
    )

    gauge_color = _score_to_color(report.total_score)

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=report.total_score,
            number=dict(suffix="/100", font=dict(size=28)),
            title=dict(
                text=f"<b>Grade: {report.grade}</b>",
                font=dict(size=20, color=_GRADE_COLOR.get(report.grade, "black")),
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="darkgray",
                          tick0=0, dtick=10),
                bar=dict(color=gauge_color, thickness=0.3),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[lo, hi], color=color.replace("0.9)", "0.2)"))
                    for lo, hi, color in _ZONE_COLORS
                ],
                threshold=dict(
                    line=dict(color="black", width=4),
                    thickness=0.7,
                    value=report.total_score,
                ),
            ),
        ),
        row=1, col=1,
    )

    comp_keys = list(_COMPONENT_LABELS.keys())
    comp_scores = [report.component_scores.get(k, 0) for k in comp_keys]
    comp_labels = [_COMPONENT_LABELS[k] for k in comp_keys]
    bar_colors = [_score_to_color(s * 4) for s in comp_scores]  # scale 25 → 100

    fig.add_trace(
        go.Bar(
            x=comp_scores,
            y=comp_labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{s:.1f}/25" for s in comp_scores],
            textposition="auto",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=520,
        title=dict(text="RD Credibility Score", font=dict(size=18)),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis2=dict(range=[0, 25], showgrid=False, title="Score (0–25 each)"),
        yaxis2=dict(showgrid=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_publication(report: CredibilityReport):
    """
    AER-style score overview panel (Matplotlib).

    Returns a figure with a large score + grade display and horizontal
    component bars.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2),
                              gridspec_kw={"width_ratios": [1, 1.6]})

    # Left panel: score circle + grade
    ax_left = axes[0]
    ax_left.axis("off")

    grade_color = _GRADE_COLOR.get(report.grade, "black")
    circle = plt.Circle((0.5, 0.52), 0.42, color=grade_color, alpha=0.15, transform=ax_left.transAxes)
    ax_left.add_patch(circle)

    ax_left.text(0.5, 0.60, f"{report.total_score:.0f}", fontsize=38,
                 ha="center", va="center", transform=ax_left.transAxes,
                 fontweight="bold", color=grade_color)
    ax_left.text(0.5, 0.32, f"Grade {report.grade}", fontsize=16,
                 ha="center", va="center", transform=ax_left.transAxes,
                 color=grade_color)
    ax_left.text(0.5, 0.12, "RD Credibility Score", fontsize=9,
                 ha="center", va="center", transform=ax_left.transAxes,
                 color="gray")

    # Right panel: horizontal component bars
    ax_right = axes[1]
    comp_keys = list(_COMPONENT_LABELS.keys())
    comp_scores = [report.component_scores.get(k, 0) for k in comp_keys]
    comp_labels = [_COMPONENT_LABELS[k] for k in comp_keys]
    bar_colors = [_GRADE_COLOR.get(report.grade, "steelblue")] * len(comp_keys)

    # Use zone colors per component
    bar_colors = []
    for s in comp_scores:
        scaled = s * 4  # 0-25 → 0-100
        if scaled < 40:
            bar_colors.append(_GRADE_COLOR["F"])
        elif scaled < 55:
            bar_colors.append(_GRADE_COLOR["D"])
        elif scaled < 70:
            bar_colors.append(_GRADE_COLOR["C"])
        elif scaled < 85:
            bar_colors.append(_GRADE_COLOR["B"])
        else:
            bar_colors.append(_GRADE_COLOR["A"])

    y_pos = range(len(comp_keys))
    bars = ax_right.barh(list(y_pos), comp_scores, color=bar_colors, height=0.6, alpha=0.85)
    ax_right.set_xlim(0, 27)
    ax_right.set_yticks(list(y_pos))
    ax_right.set_yticklabels(comp_labels, fontsize=9)
    ax_right.set_xlabel("Component score (0–25)", fontsize=9)

    for bar, score in zip(bars, comp_scores):
        ax_right.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      f"{score:.0f}", va="center", ha="left", fontsize=9)

    ax_right.axvline(25, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.grid(False)

    fig.suptitle("RD Credibility Assessment", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig
