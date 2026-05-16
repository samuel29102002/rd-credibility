"""Tests for all visualization modules."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tests.fixtures.synthetic_rd import (
    generate_rd_data,
    generate_rd_data_with_covariates,
    generate_manipulation_data,
)
from rd_credibility.diagnostics.mccrary import McCraryTest
from rd_credibility.diagnostics.covariate_balance import CovariateBalance
from rd_credibility.diagnostics.placebo_cutoffs import PlaceboTest
from rd_credibility.diagnostics.bandwidth_grid import BandwidthSensitivity
from rd_credibility.scoring.credibility import CredibilityScore
from rd_credibility.visualization import (
    rd_plot,
    density_plot,
    sensitivity_heatmap,
    covariate_grid,
    placebo_plot,
    score_gauge,
    export,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_data():
    df = generate_rd_data(n=500, tau=1.0, seed=42)
    return df["y"].values, df["x"].values


@pytest.fixture(scope="module")
def mccrary_result(base_data):
    y, x = base_data
    return McCraryTest(x).fit()


@pytest.fixture(scope="module")
def balance_result(base_data):
    y, x = base_data
    df = generate_rd_data_with_covariates(n=500, n_covariates=3, seed=42)
    covs = df[[c for c in df.columns if c.startswith("z")]]
    return CovariateBalance(x, covs, bandwidth=0.4).fit()


@pytest.fixture(scope="module")
def placebo_result(base_data):
    y, x = base_data
    return PlaceboTest(y, x, n_placebo=10).fit()


@pytest.fixture(scope="module")
def bw_result(base_data):
    y, x = base_data
    return BandwidthSensitivity(y, x, poly_orders=[1, 2]).fit()


@pytest.fixture(scope="module")
def credibility_report(base_data, mccrary_result, balance_result, placebo_result, bw_result):
    y, x = base_data
    scorer = CredibilityScore(
        mccrary_result=mccrary_result,
        balance_result=balance_result,
        bandwidth_result=bw_result,
        placebo_result=placebo_result,
    )
    return scorer.compute()


# ---------------------------------------------------------------------------
# rd_plot
# ---------------------------------------------------------------------------

class TestRdPlot:
    def test_plot_interactive_returns_figure(self, base_data):
        y, x = base_data
        fig = rd_plot.plot_interactive(y, x)
        assert isinstance(fig, go.Figure)

    def test_plot_interactive_has_scatter_traces(self, base_data):
        y, x = base_data
        fig = rd_plot.plot_interactive(y, x, n_bins=20)
        scatter_types = [t.type for t in fig.data]
        assert "scatter" in scatter_types

    def test_plot_publication_returns_figure(self, base_data):
        y, x = base_data
        fig = rd_plot.plot_publication(y, x)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_publication_axes_labels(self, base_data):
        y, x = base_data
        fig = rd_plot.plot_publication(y, x)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_plot_interactive_explicit_bandwidth(self, base_data):
        y, x = base_data
        fig = rd_plot.plot_interactive(y, x, bandwidth=0.4)
        assert isinstance(fig, go.Figure)

    def test_compute_bin_means_length(self, base_data):
        y, x = base_data
        bx, by = rd_plot._compute_bin_means(x, y, n_bins=20)
        assert len(bx) == len(by)
        assert len(bx) <= 20

    def test_fit_poly_with_ci_returns_both_sides(self, base_data):
        y, x = base_data
        fits = rd_plot._fit_poly_with_ci(y, x, cutoff=0.0, bandwidth=0.4, poly_order=1)
        assert "left" in fits and "right" in fits
        for side in ("left", "right"):
            fit = fits[side]
            assert fit is not None
            assert "x_grid" in fit and "y_pred" in fit
            assert "ci_lower" in fit and "ci_upper" in fit

    def test_ci_ordering(self, base_data):
        y, x = base_data
        fits = rd_plot._fit_poly_with_ci(y, x, cutoff=0.0, bandwidth=0.4, poly_order=1)
        for side in ("left", "right"):
            fit = fits[side]
            assert np.all(fit["ci_lower"] <= fit["ci_upper"])

    def test_plot_publication_no_top_right_spines(self, base_data):
        y, x = base_data
        fig = rd_plot.plot_publication(y, x)
        ax = fig.axes[0]
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)


# ---------------------------------------------------------------------------
# density_plot
# ---------------------------------------------------------------------------

class TestDensityPlot:
    def test_plot_interactive_returns_figure(self, mccrary_result):
        fig = density_plot.plot_interactive(mccrary_result)
        assert isinstance(fig, go.Figure)

    def test_plot_interactive_has_bars(self, mccrary_result):
        fig = density_plot.plot_interactive(mccrary_result)
        bar_traces = [t for t in fig.data if t.type == "bar"]
        assert len(bar_traces) >= 1

    def test_plot_publication_returns_figure(self, mccrary_result):
        fig = density_plot.plot_publication(mccrary_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_publication_axes_labels(self, mccrary_result):
        fig = density_plot.plot_publication(mccrary_result)
        ax = fig.axes[0]
        assert "Running" in ax.get_xlabel() or ax.get_xlabel() != ""
        plt.close(fig)

    def test_plot_publication_no_grid(self, mccrary_result):
        fig = density_plot.plot_publication(mccrary_result)
        ax = fig.axes[0]
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)


# ---------------------------------------------------------------------------
# sensitivity_heatmap
# ---------------------------------------------------------------------------

class TestSensitivityHeatmap:
    def test_plot_interactive_returns_figure(self, bw_result):
        fig = sensitivity_heatmap.plot_interactive(bw_result)
        assert isinstance(fig, go.Figure)

    def test_plot_interactive_has_heatmap(self, bw_result):
        fig = sensitivity_heatmap.plot_interactive(bw_result)
        heatmap_traces = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmap_traces) == 1

    def test_plot_publication_returns_figure(self, bw_result):
        fig = sensitivity_heatmap.plot_publication(bw_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_build_pivot_shape(self, bw_result):
        est_pivot, se_pivot = sensitivity_heatmap._build_pivot(bw_result.grid)
        assert est_pivot.shape == se_pivot.shape
        assert est_pivot.shape[0] >= 1  # at least one poly order
        assert est_pivot.shape[1] >= 2  # at least two bandwidths

    def test_baseline_stats_finite(self, bw_result):
        est, se = sensitivity_heatmap._baseline_stats(bw_result.grid, bw_result.optimal_bandwidth)
        assert np.isfinite(est)
        assert np.isfinite(se) and se > 0

    def test_plot_publication_labels(self, bw_result):
        fig = sensitivity_heatmap.plot_publication(bw_result)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


# ---------------------------------------------------------------------------
# covariate_grid
# ---------------------------------------------------------------------------

class TestCovariateGrid:
    def test_plot_interactive_returns_figure(self, balance_result):
        fig = covariate_grid.plot_interactive(balance_result)
        assert isinstance(fig, go.Figure)

    def test_plot_publication_returns_figure(self, balance_result):
        fig = covariate_grid.plot_publication(balance_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sorted_df_columns(self, balance_result):
        df = covariate_grid._sorted_df(balance_result)
        assert "covariate" in df.columns
        assert "estimate" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        assert "p_value" in df.columns

    def test_sorted_df_order(self, balance_result):
        df = covariate_grid._sorted_df(balance_result)
        if len(df) > 1:
            assert df["p_value"].is_monotonic_increasing

    def test_color_from_pvalue(self):
        assert covariate_grid._color_from_pvalue(0.01) == covariate_grid._RED
        assert covariate_grid._color_from_pvalue(0.07) == covariate_grid._ORANGE
        assert covariate_grid._color_from_pvalue(0.20) == covariate_grid._GREEN

    def test_ci_lower_le_upper(self, balance_result):
        df = covariate_grid._sorted_df(balance_result)
        assert (df["ci_lower"] <= df["ci_upper"]).all()

    def test_plot_publication_no_top_right_spines(self, balance_result):
        fig = covariate_grid.plot_publication(balance_result)
        ax = fig.axes[0]
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)


# ---------------------------------------------------------------------------
# placebo_plot
# ---------------------------------------------------------------------------

class TestPlaceboPlot:
    def test_plot_interactive_returns_figure(self, placebo_result):
        fig = placebo_plot.plot_interactive(placebo_result)
        assert isinstance(fig, go.Figure)

    def test_plot_publication_returns_figure(self, placebo_result):
        fig = placebo_plot.plot_publication(placebo_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_significance_mask_shape(self, placebo_result):
        mask = placebo_plot._significance_mask(placebo_result)
        assert mask.shape == placebo_result.placebo_estimates.shape
        assert mask.dtype == bool

    def test_significance_mask_no_nan_entries(self, placebo_result):
        mask = placebo_plot._significance_mask(placebo_result)
        # Where SE <= 0 or estimate is NaN, mask must be False
        bad = np.isnan(placebo_result.placebo_estimates) | (placebo_result.placebo_ses <= 0)
        assert not mask[bad].any()

    def test_plot_publication_shows_true_estimate(self, placebo_result):
        fig = placebo_plot.plot_publication(placebo_result)
        ax = fig.axes[0]
        vlines = [line for line in ax.get_lines()
                  if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
        assert len(vlines) >= 1
        plt.close(fig)

    def test_plot_publication_no_top_right_spines(self, placebo_result):
        fig = placebo_plot.plot_publication(placebo_result)
        ax = fig.axes[0]
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)


# ---------------------------------------------------------------------------
# score_gauge
# ---------------------------------------------------------------------------

class TestScoreGauge:
    def test_plot_interactive_returns_figure(self, credibility_report):
        fig = score_gauge.plot_interactive(credibility_report)
        assert isinstance(fig, go.Figure)

    def test_plot_interactive_has_indicator(self, credibility_report):
        fig = score_gauge.plot_interactive(credibility_report)
        types = [t.type for t in fig.data]
        assert "indicator" in types

    def test_plot_interactive_has_bar(self, credibility_report):
        fig = score_gauge.plot_interactive(credibility_report)
        types = [t.type for t in fig.data]
        assert "bar" in types

    def test_plot_publication_returns_figure(self, credibility_report):
        fig = score_gauge.plot_publication(credibility_report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_score_to_color_zones(self):
        assert score_gauge._score_to_color(20) == score_gauge._ZONE_COLORS[0][2]
        assert score_gauge._score_to_color(95) == score_gauge._ZONE_COLORS[-1][2]

    def test_plot_publication_two_axes(self, credibility_report):
        fig = score_gauge.plot_publication(credibility_report)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_score_range(self, credibility_report):
        assert 0 <= credibility_report.total_score <= 100

    def test_grade_is_letter(self, credibility_report):
        assert credibility_report.grade in ("A", "B", "C", "D", "F")


# ---------------------------------------------------------------------------
# export (save_publication_panel)
# ---------------------------------------------------------------------------

class TestExport:
    def test_save_publication_panel_returns_figure(self, tmp_path, base_data,
                                                     mccrary_result, balance_result,
                                                     placebo_result):
        y, x = base_data
        all_results = {
            "y": y, "x": x, "cutoff": 0.0,
            "bandwidth": 0.4, "poly_order": 1,
            "mccrary": mccrary_result,
            "balance": balance_result,
            "placebo": placebo_result,
        }
        out = str(tmp_path / "panel")
        fig = export.save_publication_panel(all_results, output_path=out)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_publication_panel_writes_pdf(self, tmp_path, base_data,
                                                mccrary_result, balance_result,
                                                placebo_result):
        y, x = base_data
        all_results = {
            "y": y, "x": x, "cutoff": 0.0,
            "bandwidth": 0.4, "poly_order": 1,
            "mccrary": mccrary_result,
            "balance": balance_result,
            "placebo": placebo_result,
        }
        out = str(tmp_path / "panel")
        export.save_publication_panel(all_results, output_path=out)
        assert (tmp_path / "panel.pdf").exists()

    def test_save_publication_panel_writes_png(self, tmp_path, base_data,
                                                mccrary_result, balance_result,
                                                placebo_result):
        y, x = base_data
        all_results = {
            "y": y, "x": x, "cutoff": 0.0,
            "bandwidth": 0.4, "poly_order": 1,
            "mccrary": mccrary_result,
            "balance": balance_result,
            "placebo": placebo_result,
        }
        out = str(tmp_path / "panel")
        export.save_publication_panel(all_results, output_path=out)
        assert (tmp_path / "panel.png").exists()

    def test_save_publication_panel_four_subplots(self, tmp_path, base_data,
                                                   mccrary_result, balance_result,
                                                   placebo_result):
        y, x = base_data
        all_results = {
            "y": y, "x": x, "cutoff": 0.0,
            "bandwidth": 0.4, "poly_order": 1,
            "mccrary": mccrary_result,
            "balance": balance_result,
            "placebo": placebo_result,
        }
        out = str(tmp_path / "panel2")
        fig = export.save_publication_panel(all_results, output_path=out)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_save_publication_panel_auto_bandwidth(self, tmp_path, base_data,
                                                    mccrary_result, balance_result,
                                                    placebo_result):
        y, x = base_data
        all_results = {
            "y": y, "x": x, "cutoff": 0.0,
            "mccrary": mccrary_result,
            "balance": balance_result,
            "placebo": placebo_result,
        }
        out = str(tmp_path / "panel3")
        fig = export.save_publication_panel(all_results, output_path=out)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_rd_onto_ax(self, base_data):
        y, x = base_data
        fig, ax = plt.subplots()
        export._draw_rd_onto_ax(ax, y, x, cutoff=0.0, bandwidth=0.4, poly_order=1)
        assert len(ax.collections) >= 1  # scatter points drawn
        plt.close(fig)

    def test_draw_density_onto_ax(self, mccrary_result):
        fig, ax = plt.subplots()
        export._draw_density_onto_ax(ax, mccrary_result, cutoff=0.0)
        assert len(ax.patches) >= 1  # bars drawn
        plt.close(fig)

    def test_draw_balance_onto_ax(self, balance_result):
        fig, ax = plt.subplots()
        export._draw_balance_onto_ax(ax, balance_result)
        plt.close(fig)

    def test_draw_placebo_onto_ax(self, placebo_result):
        fig, ax = plt.subplots()
        export._draw_placebo_onto_ax(ax, placebo_result)
        plt.close(fig)
