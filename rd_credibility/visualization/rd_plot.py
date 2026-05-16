"""RD scatter + fitted polynomial plot."""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.kernels import get_kernel
from rd_credibility.estimation.rdrobust import RDEstimator

_LEFT_COLOR = "#4682B4"   # steelblue
_RIGHT_COLOR = "#E8735A"  # coral
_LEFT_FILL = "rgba(70,130,180,0.15)"
_RIGHT_FILL = "rgba(232,115,90,0.15)"


def _compute_bin_means(x, y, n_bins):
    """Return (bin_x, bin_y) arrays using evenly-spaced bins."""
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bx, by = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & (x < hi) if i < n_bins - 1 else (x >= lo) & (x <= hi)
        if mask.sum() > 0:
            bx.append(x[mask].mean())
            by.append(y[mask].mean())
    return np.array(bx), np.array(by)


def _fit_poly_with_ci(y, x, cutoff, bandwidth, poly_order, kernel="triangular"):
    """
    Fit local polynomial on each side and return a fine-grid prediction.

    Returns
    -------
    dict with keys 'left' and 'right', each containing
    {'x_grid', 'y_pred', 'ci_lower', 'ci_upper'}.
    """
    kernel_fn = get_kernel(kernel)
    xc = x - cutoff
    results = {}

    for side in ("left", "right"):
        if side == "left":
            mask = (xc >= -bandwidth) & (xc < 0)
            x_grid = np.linspace(cutoff - bandwidth, cutoff, 200)
        else:
            mask = (xc >= 0) & (xc <= bandwidth)
            x_grid = np.linspace(cutoff, cutoff + bandwidth, 200)

        xs = xc[mask]
        ys = y[mask]
        if len(xs) < poly_order + 2:
            results[side] = None
            continue

        w = kernel_fn(xs / bandwidth)
        X = np.column_stack([xs**p for p in range(poly_order + 1)])
        Xw = X * w[:, None]
        XtWX = Xw.T @ X
        XtWy = Xw.T @ ys

        try:
            inv = np.linalg.inv(XtWX)
            coeffs = inv @ XtWy
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
            inv = np.linalg.pinv(XtWX)

        residuals = ys - X @ coeffs
        Xwe = Xw * residuals[:, None]
        meat = Xwe.T @ Xwe
        n, p = X.shape
        V = inv @ meat @ inv * (n / max(n - p, 1))

        xgc = x_grid - cutoff
        X_pred = np.column_stack([xgc**k for k in range(poly_order + 1)])
        y_pred = X_pred @ coeffs
        var_pred = np.maximum(np.einsum("ij,jk,ik->i", X_pred, V, X_pred), 0)
        ci_w = 1.96 * np.sqrt(var_pred)

        results[side] = {
            "x_grid": x_grid,
            "y_pred": y_pred,
            "ci_lower": y_pred - ci_w,
            "ci_upper": y_pred + ci_w,
        }

    return results


def _run_rd(y, x, cutoff, bandwidth, poly_order):
    bw = bandwidth or mse_optimal_bandwidth(y, x, cutoff)
    result = RDEstimator(y, x, cutoff=cutoff, bandwidth=bw, poly_order=poly_order).fit()
    return bw, result


def plot_interactive(
    y, x, cutoff=0.0, bandwidth=None, poly_order=1, n_bins=30, show_ci=True
):
    """
    Interactive RD scatter + fitted polynomial plot (Plotly).

    Parameters
    ----------
    y, x : array_like
    cutoff : float
    bandwidth : float or None
    poly_order : int
    n_bins : int
    show_ci : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    bw, rd_result = _run_rd(y, x, cutoff, bandwidth, poly_order)
    fits = _fit_poly_with_ci(y, x, cutoff, bw, poly_order)

    bx, by = _compute_bin_means(x, y, n_bins)
    left_mask = bx < cutoff
    right_mask = bx >= cutoff

    fig = go.Figure()

    # Bin scatter
    for mask, color, name in [
        (left_mask, _LEFT_COLOR, "Left of cutoff"),
        (right_mask, _RIGHT_COLOR, "Right of cutoff"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=bx[mask], y=by[mask],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.85),
                name=name,
                showlegend=True,
            )
        )

    # Fitted curves + CI
    fill_colors = {"left": _LEFT_FILL, "right": _RIGHT_FILL}
    for side, color, show_legend in [("left", _LEFT_COLOR, False), ("right", _RIGHT_COLOR, False)]:
        fit = fits.get(side)
        if fit is None:
            continue
        xg, yp = fit["x_grid"], fit["y_pred"]

        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xg, xg[::-1]]),
                    y=np.concatenate([fit["ci_upper"], fit["ci_lower"][::-1]]),
                    fill="toself",
                    fillcolor=fill_colors[side],
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{side} CI",
                )
            )
        fig.add_trace(
            go.Scatter(x=xg, y=yp, mode="lines",
                       line=dict(color=color, width=2.5),
                       showlegend=show_legend)
        )

    # Cutoff line
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text="Cutoff",
        annotation_position="top right",
    )

    est, se = rd_result.estimate, rd_result.se
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.97, y=0.05,
        text=f"<b>Estimate:</b> {est:.3f} ± {se:.3f}",
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=12),
    )

    fig.update_layout(
        title=dict(text="Regression Discontinuity", font=dict(size=16)),
        xaxis_title="Running variable",
        yaxis_title="Outcome",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def plot_publication(
    y, x, cutoff=0.0, bandwidth=None, poly_order=1, n_bins=30, show_ci=True
):
    """
    AER-style publication RD plot (Matplotlib).

    Returns
    -------
    matplotlib.figure.Figure
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    bw, rd_result = _run_rd(y, x, cutoff, bandwidth, poly_order)
    fits = _fit_poly_with_ci(y, x, cutoff, bw, poly_order)

    bx, by = _compute_bin_means(x, y, n_bins)
    left_mask = bx < cutoff
    right_mask = bx >= cutoff

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    plt.rcParams.update({"font.family": "serif"})

    ax.scatter(bx[left_mask], by[left_mask], color=_LEFT_COLOR, s=30, zorder=3, alpha=0.85)
    ax.scatter(bx[right_mask], by[right_mask], color=_RIGHT_COLOR, s=30, zorder=3, alpha=0.85)

    for side, color in [("left", _LEFT_COLOR), ("right", _RIGHT_COLOR)]:
        fit = fits.get(side)
        if fit is None:
            continue
        ax.plot(fit["x_grid"], fit["y_pred"], color=color, linewidth=2.0, zorder=4)
        if show_ci:
            ax.fill_between(
                fit["x_grid"], fit["ci_lower"], fit["ci_upper"],
                color=color, alpha=0.15, zorder=2
            )

    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2, zorder=5)

    left_patch = mpatches.Patch(color=_LEFT_COLOR, label="Left of cutoff")
    right_patch = mpatches.Patch(color=_RIGHT_COLOR, label="Right of cutoff")
    ax.legend(handles=[left_patch, right_patch], frameon=False, fontsize=9)

    est, se = rd_result.estimate, rd_result.se
    ax.text(
        0.97, 0.05,
        f"Estimate: {est:.3f} ± {se:.3f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("Running variable", fontsize=10)
    ax.set_ylabel("Outcome", fontsize=10)
    ax.set_title("Regression Discontinuity", fontsize=11)
    fig.tight_layout()
    return fig
