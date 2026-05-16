"""Publication-quality panel export for RD diagnostics."""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rd_credibility.visualization import (
    covariate_grid,
    density_plot,
    placebo_plot,
    rd_plot,
)


def save_publication_panel(all_results: dict, output_path: str, format: str = "pdf"):
    """
    Combine all four core diagnostic plots into a 2×2 publication panel.

    The panel layout is:
        ┌─────────────────┬─────────────────┐
        │   RD Scatter    │ Density (McCrary)│
        ├─────────────────┼─────────────────┤
        │ Covariate Bal.  │   Placebo Test  │
        └─────────────────┴─────────────────┘

    Parameters
    ----------
    all_results : dict
        Must contain:
          - 'y'              : np.ndarray
          - 'x'              : np.ndarray
          - 'cutoff'         : float
          - 'bandwidth'      : float (optional, None → auto)
          - 'poly_order'     : int   (default 1)
          - 'mccrary'        : McCraryResult
          - 'balance'        : CovariateBalanceResult
          - 'placebo'        : PlaceboResult
    output_path : str
        Output file path WITHOUT extension (e.g., "figures/panel").
        Both PDF and PNG are always written.
    format : str
        Primary format — 'pdf' or 'png'. Default 'pdf'.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled panel figure.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    y = all_results["y"]
    x = all_results["x"]
    cutoff = float(all_results.get("cutoff", 0.0))
    bandwidth = all_results.get("bandwidth", None)
    poly_order = int(all_results.get("poly_order", 1))

    mccrary_res = all_results["mccrary"]
    balance_res = all_results["balance"]
    placebo_res = all_results["placebo"]

    fig = plt.figure(figsize=(6.5, 5.5))
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
        left=0.08, right=0.97,
        top=0.94, bottom=0.08,
    )

    panel_axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # --- Panel A: RD scatter ---
    _draw_rd_onto_ax(panel_axes[0], y, x, cutoff, bandwidth, poly_order)
    panel_axes[0].set_title("(A) RD Estimate", fontsize=10, loc="left", pad=4)

    # --- Panel B: Density ---
    _draw_density_onto_ax(panel_axes[1], mccrary_res, cutoff)
    panel_axes[1].set_title("(B) Density Continuity", fontsize=10, loc="left", pad=4)

    # --- Panel C: Covariate balance ---
    _draw_balance_onto_ax(panel_axes[2], balance_res)
    panel_axes[2].set_title("(C) Covariate Balance", fontsize=10, loc="left", pad=4)

    # --- Panel D: Placebo ---
    _draw_placebo_onto_ax(panel_axes[3], placebo_res)
    panel_axes[3].set_title("(D) Placebo Cutoff Test", fontsize=10, loc="left", pad=4)

    for ax in panel_axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        save_path = str(out) + f".{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")

    return fig


# ---------------------------------------------------------------------------
# Internal draw-onto-axis helpers (avoids re-creating full figures)
# ---------------------------------------------------------------------------


def _apply_aer(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _draw_rd_onto_ax(ax, y, x, cutoff, bandwidth, poly_order, n_bins=20):
    """Draw RD scatter + fit onto an existing axes."""
    import numpy as np
    from rd_credibility.visualization.rd_plot import (
        _compute_bin_means, _fit_poly_with_ci, _run_rd
    )

    _LEFT = "#4682B4"
    _RIGHT = "#E8735A"

    bw, rd_res = _run_rd(y, x, cutoff, bandwidth, poly_order)
    fits = _fit_poly_with_ci(y, x, cutoff, bw, poly_order)
    bx, by = _compute_bin_means(x, y, n_bins)

    ax.scatter(bx[bx < cutoff], by[bx < cutoff], color=_LEFT, s=12, alpha=0.8, zorder=3)
    ax.scatter(bx[bx >= cutoff], by[bx >= cutoff], color=_RIGHT, s=12, alpha=0.8, zorder=3)

    for side, color in [("left", _LEFT), ("right", _RIGHT)]:
        fit = fits.get(side)
        if fit:
            ax.plot(fit["x_grid"], fit["y_pred"], color=color, linewidth=1.5, zorder=4)
            ax.fill_between(fit["x_grid"], fit["ci_lower"], fit["ci_upper"],
                            color=color, alpha=0.12, zorder=2)

    ax.axvline(cutoff, color="black", linestyle="--", linewidth=0.9)
    ax.text(0.97, 0.04,
            f"Est: {rd_res.estimate:.3f} ± {rd_res.se:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax.set_xlabel("Running variable", fontsize=8)
    ax.set_ylabel("Outcome", fontsize=8)


def _draw_density_onto_ax(ax, r, cutoff):
    """Draw McCrary density plot onto an existing axes."""
    import numpy as np
    _LEFT = "#4682B4"
    _RIGHT = "#E8735A"

    bc = r.bin_centers
    lm = bc < cutoff
    rm = bc >= cutoff
    s = (bc[1] - bc[0]) if len(bc) > 1 else 0.05
    dens = r.bin_counts / (r.bin_counts.sum() * s) if r.bin_counts.sum() > 0 else r.bin_counts

    ax.bar(bc[lm], dens[lm], width=s * 0.9, color=_LEFT, alpha=0.45)
    ax.bar(bc[rm], dens[rm], width=s * 0.9, color=_RIGHT, alpha=0.45)

    if len(r.fitted_left) > 0:
        ax.plot(bc[lm], r.fitted_left, color=_LEFT, linewidth=1.5)
    if len(r.fitted_right) > 0:
        ax.plot(bc[rm], r.fitted_right, color=_RIGHT, linewidth=1.5)

    ax.axvline(cutoff, color="black", linestyle="--", linewidth=0.9)
    p_str = f"p={r.p_value:.3f}" if r.p_value >= 0.001 else "p<0.001"
    ax.text(0.03, 0.96, p_str, transform=ax.transAxes,
            va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax.set_xlabel("Running variable", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)


def _draw_balance_onto_ax(ax, balance_result):
    """Draw covariate balance dot plot onto an existing axes."""
    import numpy as np
    from rd_credibility.visualization.covariate_grid import _sorted_df, _color_from_pvalue

    df = _sorted_df(balance_result)
    if df.empty:
        ax.text(0.5, 0.5, "No covariates", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    for i, (_, row) in enumerate(df.iterrows()):
        color = _color_from_pvalue(row["p_value"])
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], color=color, linewidth=1.5)
        ax.scatter(row["estimate"], i, color=color, s=18, zorder=5)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["covariate"].tolist(), fontsize=7.5)
    ax.set_xlabel("RD Estimate", fontsize=8)


def _draw_placebo_onto_ax(ax, r):
    """Draw placebo histogram + rug onto an existing axes."""
    import numpy as np
    from rd_credibility.visualization.placebo_plot import _significance_mask

    _INSIG = "#4682B4"
    _SIG = "#D62728"
    _TRUE = "#2CA02C"

    estimates = r.placebo_estimates
    sig_mask = _significance_mask(r)
    valid = ~np.isnan(estimates)

    if valid.sum() >= 3:
        ax.hist(estimates[valid], bins=max(5, int(valid.sum() // 3)),
                color=_INSIG, alpha=0.5, edgecolor="white")
        for est, is_sig in zip(estimates[valid], sig_mask[valid]):
            ax.axvline(est, ymin=0, ymax=0.07,
                       color=_SIG if is_sig else _INSIG, linewidth=1.2)

    ax.axvline(r.true_estimate, color=_TRUE, linewidth=1.8,
               label=f"True: {r.true_estimate:.3f}")
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    n_sig, n_total = int(sig_mask.sum()), int(valid.sum())
    ax.text(0.97, 0.96,
            f"{n_sig}/{n_total} sig.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlabel("Placebo Estimate", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
