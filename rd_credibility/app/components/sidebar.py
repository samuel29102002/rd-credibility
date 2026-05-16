"""Sidebar: dataset selector, parameter controls, learning mode toggle."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st

from rd_credibility.app.components.data_loader import (
    DEFAULT_CUTOFFS,
    covariate_columns,
    load_builtin,
)

_BUILTIN = ["Lee 2008 (Electoral)", "Maimonides Rule", "Thistlethwaite 1960"]
_KERNELS = ["Triangular", "Uniform", "Epanechnikov"]
_KERNEL_MAP = {"Triangular": "triangular", "Uniform": "uniform", "Epanechnikov": "epanechnikov"}


def render_sidebar() -> dict:
    """
    Render sidebar controls and return a dict of configuration values:
      df, y_col, x_col, cov_cols, cutoff, bandwidth (None = MSE-optimal),
      poly_order, kernel, learning_mode
    """
    st.sidebar.title("Controls")

    # --- Dataset ---
    st.sidebar.subheader("Dataset")
    options = _BUILTIN + ["Upload CSV"]
    dataset_name = st.sidebar.selectbox("Select dataset", options, key="dataset_name")

    df = None
    y_col = "y"
    x_col = "x"
    cov_cols = []

    if dataset_name == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is None:
            st.sidebar.info("Upload a CSV to continue.")
            return _empty_config()

        df = pd.read_csv(uploaded)
        cols = df.columns.tolist()
        if len(cols) < 2:
            st.sidebar.error("CSV must have at least 2 columns.")
            return _empty_config()

        st.sidebar.markdown("**Column mapping**")
        x_col = st.sidebar.selectbox("Running variable (X)", cols, key="csv_x")
        y_col = st.sidebar.selectbox("Outcome (Y)", cols,
                                      index=min(1, len(cols) - 1), key="csv_y")
        remaining = [c for c in cols if c not in (x_col, y_col)]
        cov_cols = st.sidebar.multiselect("Covariates (optional)", remaining, key="csv_covs")

        default_cutoff = 0.0
    else:
        df = load_builtin(dataset_name)
        cov_cols = covariate_columns(df)
        default_cutoff = DEFAULT_CUTOFFS.get(dataset_name, 0.0)

    # --- Cutoff ---
    st.sidebar.subheader("RD Parameters")
    x_min = float(df[x_col].min()) if df is not None else -1.0
    x_max = float(df[x_col].max()) if df is not None else 1.0
    cutoff = st.sidebar.number_input(
        "Cutoff", value=default_cutoff, step=0.01,
        min_value=x_min, max_value=x_max, key="cutoff"
    )

    # --- Bandwidth ---
    bw_mode = st.sidebar.radio(
        "Bandwidth selection",
        ["MSE-Optimal (CCT)", "Manual"],
        key="bw_mode",
    )
    bandwidth = None
    if bw_mode == "Manual":
        x_range = x_max - x_min
        bandwidth = st.sidebar.slider(
            "Bandwidth (h)",
            min_value=round(x_range * 0.05, 3),
            max_value=round(x_range * 0.70, 3),
            value=round(x_range * 0.25, 3),
            step=round(x_range * 0.01, 3),
            key="bw_manual",
        )

    # --- Polynomial order ---
    poly_order = st.sidebar.select_slider(
        "Polynomial order", options=[1, 2, 3], value=1, key="poly_order"
    )

    # --- Kernel ---
    kernel_label = st.sidebar.selectbox("Kernel", _KERNELS, key="kernel")
    kernel = _KERNEL_MAP[kernel_label]

    # --- Learning mode ---
    st.sidebar.subheader("Options")
    learning_mode = st.sidebar.toggle("Learning Mode", value=False, key="learning_mode")
    if learning_mode:
        st.sidebar.caption(
            "Explanatory panels will appear below each diagnostic, "
            "describing what the test checks and what the result means."
        )

    return {
        "df": df,
        "y_col": y_col,
        "x_col": x_col,
        "cov_cols": cov_cols,
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "poly_order": poly_order,
        "kernel": kernel,
        "learning_mode": learning_mode,
        "dataset_name": dataset_name,
    }


def _empty_config() -> dict:
    return {
        "df": None, "y_col": "y", "x_col": "x", "cov_cols": [],
        "cutoff": 0.0, "bandwidth": None, "poly_order": 1,
        "kernel": "triangular", "learning_mode": False, "dataset_name": None,
    }
