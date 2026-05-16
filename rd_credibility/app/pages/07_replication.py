"""Replication Mode — audit a published RD specification against our full diagnostic suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Replication Mode · RD Dashboard", layout="wide")

from rd_credibility.app.components.replication import (
    PreloadedExample,
    ReplicationAudit,
    get_preloaded_examples,
    run_replication_audit,
)
from rd_credibility.app.components.style import inject_css, page_desc, badge, kpi_card
from rd_credibility.visualization import density_plot, placebo_plot, sensitivity_heatmap

inject_css()

# ── Additional CSS for the audit panel ────────────────────────────────────────
st.markdown("""
<style>
.audit-section {
    background: #fafafa;
    border: 1px solid #e0e0e8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.audit-section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a1a2e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}
.verdict-robust  { background:#e8f5e9; border-left:5px solid #27ae60; padding:1rem 1.2rem; border-radius:6px; }
.verdict-fragile { background:#fff8e1; border-left:5px solid #f39c12; padding:1rem 1.2rem; border-radius:6px; }
.verdict-problematic { background:#fdecea; border-left:5px solid #c0392b; padding:1rem 1.2rem; border-radius:6px; }
.dim-label { font-size:0.78rem; color:#888899; text-transform:uppercase; letter-spacing:0.06em; }
.dim-value { font-size:1.35rem; font-weight:700; color:#1a1a2e; }
.check-pass { color:#27ae60; font-weight:600; }
.check-warn { color:#f39c12; font-weight:600; }
.check-fail { color:#c0392b; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("Replication Mode")
page_desc(
    "Upload data from a published RD paper and audit whether the reported specification "
    "is reproducible, robust to bandwidth choice, and supported by overlooked diagnostics."
)

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Choose mode",
    ["Pre-loaded Examples", "Upload Your Own Paper"],
    horizontal=True,
    key="replication_mode",
)

st.markdown("---")

# ── Helper: get data + spec from user or from pre-loaded example ──────────────

example: PreloadedExample | None = None
audit_ready = False
y = x = cov_cols = cov_data = None
cutoff = reported_bw = reported_est = reported_se = None
poly_order = 1

if mode == "Pre-loaded Examples":
    examples = get_preloaded_examples()
    example_name = st.selectbox(
        "Select pre-loaded example",
        list(examples.keys()),
        key="preloaded_example_name",
    )
    example = examples[example_name]

    # Show description card
    col_desc, col_ref = st.columns([2, 1])
    with col_desc:
        st.markdown(f"**{example.name}**")
        st.markdown(example.description)
    with col_ref:
        st.markdown("**Paper reference**")
        st.caption(example.paper_ref)
        st.markdown(
            f'Expected verdict: {badge("pass" if example.expected_verdict == "Robust" else ("warn" if example.expected_verdict == "Fragile" else "fail"), example.expected_verdict)}',
            unsafe_allow_html=True,
        )

    st.markdown("**Reported specification**")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Cutoff", f"{example.cutoff:.3f}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Bandwidth", f"{example.reported_bandwidth:.4f}"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Estimate", f"{example.reported_estimate:.4f}"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Reported SE", f"{example.reported_se:.4f}"), unsafe_allow_html=True)

    y = example.df[example.y_col].values
    x = example.df[example.x_col].values
    cov_cols = example.cov_cols
    cov_data = example.df[cov_cols].values if cov_cols else None
    cutoff = example.cutoff
    reported_bw = example.reported_bandwidth
    reported_est = example.reported_estimate
    reported_se = example.reported_se
    audit_ready = True

else:
    # ── Upload flow ────────────────────────────────────────────────────────────
    st.subheader("Step 1 — Upload CSV")
    uploaded = st.file_uploader("CSV file", type=["csv"], key="repl_upload")

    if uploaded is None:
        st.info("Upload a CSV file to continue.")
        st.stop()

    df_upload = pd.read_csv(uploaded)
    cols = df_upload.columns.tolist()

    st.subheader("Step 2 — Map Columns")
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("Running variable (X)", cols, key="repl_x")
    y_col = c2.selectbox("Outcome (Y)", cols, index=min(1, len(cols) - 1), key="repl_y")
    remaining = [c for c in cols if c not in (x_col, y_col)]
    cov_cols_sel = st.multiselect("Pre-treatment covariates (optional)", remaining, key="repl_covs")

    st.subheader("Step 3 — Enter Reported Specification")
    c1, c2, c3, c4 = st.columns(4)
    cutoff = c1.number_input("Cutoff", value=0.0, step=0.01, key="repl_cutoff")
    reported_bw = c2.number_input("Reported bandwidth", value=0.3, min_value=0.001, step=0.01, key="repl_bw")
    reported_est = c3.number_input("Reported estimate", value=0.5, step=0.01, key="repl_est")
    reported_se = c4.number_input("Reported SE", value=0.05, min_value=0.0001, step=0.005, key="repl_se")

    poly_order = st.select_slider("Polynomial order", options=[1, 2, 3], value=1, key="repl_poly")

    if st.button("Run Replication Audit", type="primary", key="repl_run"):
        y = df_upload[x_col].values  # Note: keep as x_col to get x
        x = df_upload[x_col].values
        y = df_upload[y_col].values
        cov_cols = cov_cols_sel
        cov_data = df_upload[cov_cols].values if cov_cols else None
        audit_ready = True
    else:
        st.stop()

if not audit_ready:
    st.stop()

# ── Run the audit ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_audit(
    y_arr, x_arr, cov_cols_t, cov_data_arr,
    cutoff_, reported_bw_, reported_est_, reported_se_, poly_
):
    return run_replication_audit(
        y=y_arr, x=x_arr,
        cov_cols=list(cov_cols_t),
        cov_data=cov_data_arr,
        cutoff=cutoff_,
        reported_bandwidth=reported_bw_,
        reported_estimate=reported_est_,
        reported_se=reported_se_,
        poly_order=poly_,
    )

with st.spinner("Running replication audit…"):
    audit: ReplicationAudit = _cached_audit(
        y, x,
        tuple(cov_cols) if cov_cols else (),
        cov_data,
        cutoff, reported_bw, reported_est, reported_se, poly_order,
    )

# ─────────────────────────────────────────────────────────────────────────────
# REPLICATION AUDIT PANEL
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Replication Audit Panel")

# ── Verdict banner ────────────────────────────────────────────────────────────
_VERDICT_ICONS = {"Robust": "✅", "Fragile": "⚠️", "Problematic": "❌"}
_VERDICT_CLASS = {"Robust": "verdict-robust", "Fragile": "verdict-fragile", "Problematic": "verdict-problematic"}
_VERDICT_STATUS = {"Robust": "pass", "Fragile": "warn", "Problematic": "fail"}

icon = _VERDICT_ICONS[audit.verdict]
vcls = _VERDICT_CLASS[audit.verdict]

verdict_html = f'<div class="{vcls}"><span style="font-size:1.5rem;font-weight:700;">{icon} {audit.verdict}</span>'
if audit.verdict == "Robust":
    verdict_html += "<br>The reported specification is reproducible, robust to bandwidth choice, and all diagnostics pass."
elif audit.verdict == "Fragile":
    verdict_html += "<br>The estimate reproduces, but the specification has fragility concerns that should be addressed before publication."
else:
    verdict_html += "<br>This specification has fundamental problems that undermine the validity of the reported findings."
verdict_html += "</div>"
st.markdown(verdict_html, unsafe_allow_html=True)

st.markdown("")

# Reason + positive bullets
two_col = st.columns(2)
with two_col[0]:
    if audit.verdict_reasons:
        st.markdown("**Issues identified**")
        for r in audit.verdict_reasons:
            prefix = "❌" if audit.verdict == "Problematic" else "⚠️"
            st.markdown(f"{prefix} {r}")

with two_col[1]:
    if audit.verdict_positives:
        st.markdown("**What checks out**")
        for p in audit.verdict_positives:
            st.markdown(f"✅ {p}")

st.markdown("---")

# ── Four audit dimensions ─────────────────────────────────────────────────────
dim1, dim2 = st.columns(2)

# 1. Reproduced Estimate
with dim1:
    est = audit.estimate
    est_status_map = {"reproduced": "pass", "marginal": "warn", "failed": "fail"}
    st.markdown('<div class="audit-section">', unsafe_allow_html=True)
    st.markdown('<div class="audit-section-title">1 · Reproduced Estimate</div>', unsafe_allow_html=True)
    st.markdown(
        badge(est_status_map[est.status], est.status.upper()),
        unsafe_allow_html=True,
    )
    st.markdown("")

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Reported", f"{est.reported_estimate:.4f}", help="Point estimate from the paper")
    _c2.metric("Our estimate", f"{est.our_estimate:.4f}", delta=f"{est.diff:+.4f}")
    _c3.metric("Difference", f"{est.diff_in_se:.2f} SE", help="Difference scaled by reported SE")

    if est.status == "reproduced":
        st.success(f"Estimates agree within {est.diff_in_se:.2f} reported standard errors.")
    elif est.status == "marginal":
        st.warning(f"Estimates differ by {est.diff_in_se:.2f} SEs — marginal reproduction.")
    else:
        st.error(
            f"Estimates differ by {est.diff_in_se:.1f} SEs. Possible causes: different "
            "software, kernel, sample restriction, or data transformation."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# 2. Bandwidth Choice
with dim2:
    bw = audit.bandwidth
    st.markdown('<div class="audit-section">', unsafe_allow_html=True)
    st.markdown('<div class="audit-section-title">2 · Bandwidth Choice</div>', unsafe_allow_html=True)
    bw_status_map = {"optimal": "pass", "acceptable": "warn", "suspicious": "fail"}
    st.markdown(
        badge(bw_status_map[bw.status], bw.status.upper()),
        unsafe_allow_html=True,
    )
    st.markdown("")

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Reported BW", f"{bw.reported_bw:.4f}")
    _c2.metric("MSE-optimal BW", f"{bw.mse_optimal_bw:.4f}")
    _c3.metric("Ratio", f"{bw.ratio:.2f}×")

    sr = bw.stable_region
    st.markdown(f"**Stable region:** [{sr[0]:.4f}, {sr[1]:.4f}]")
    if bw.in_stable_region:
        st.success("Reported bandwidth is inside the stable region.")
    else:
        st.error(
            f"Reported bandwidth ({bw.reported_bw:.4f}) is **outside** the stable region "
            f"[{sr[0]:.4f}, {sr[1]:.4f}]. Small bandwidth changes may flip results."
        )
    st.markdown("</div>", unsafe_allow_html=True)

dim3, dim4 = st.columns(2)

# 3. Specification Fragility
with dim3:
    frag = audit.fragility
    frag_status_map = {"stable": "pass", "moderate": "warn", "fragile": "fail"}
    st.markdown('<div class="audit-section">', unsafe_allow_html=True)
    st.markdown('<div class="audit-section-title">3 · Specification Fragility</div>', unsafe_allow_html=True)
    st.markdown(
        badge(frag_status_map[frag.status], frag.status.upper()),
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.metric(
        "Fragility score",
        f"{frag.fragility_score:.0%}",
        help=f"{frag.n_stable} of {frag.n_nearby} nearby bandwidths give consistent estimates",
    )
    st.caption(
        f"{frag.n_stable}/{frag.n_nearby} bandwidths in the neighbourhood "
        f"[{bw.reported_bw * 0.40:.3f}–{bw.reported_bw * 2.50:.3f}] "
        "give estimates within ±1.96 median SE of the median."
    )

    if frag.status == "stable":
        st.success("Estimates are consistent across the neighbourhood of the reported bandwidth.")
    elif frag.status == "moderate":
        st.warning("Moderate sensitivity — some nearby bandwidths give noticeably different estimates.")
    else:
        st.error(
            "High sensitivity — the estimate changes substantially with small bandwidth perturbations. "
            "The reported specification may be a cherry-picked point in bandwidth space."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# 4. Overlooked Diagnostics
with dim4:
    diag = audit.diagnostics
    st.markdown('<div class="audit-section">', unsafe_allow_html=True)
    st.markdown('<div class="audit-section-title">4 · Overlooked Diagnostics</div>', unsafe_allow_html=True)

    icon_map = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

    st.markdown(
        f'{icon_map[diag.mccrary_status]} **McCrary (density):** {diag.mccrary_msg}'
    )
    if diag.has_covariates:
        st.markdown(
            f'{icon_map[diag.balance_status]} **Covariate balance:** {diag.balance_msg}'
        )
    else:
        st.markdown("ℹ️ **Covariate balance:** No covariates provided.")
    st.markdown(
        f'{icon_map[diag.placebo_status]} **Placebo tests:** {diag.placebo_msg}'
    )

    if not diag.any_warning:
        st.success("All diagnostic checks pass — no overlooked concerns.")
    else:
        warnings = []
        if diag.mccrary_status != "pass":
            warnings.append("McCrary density test")
        if diag.balance_status not in ("pass",) and diag.has_covariates:
            warnings.append("covariate balance")
        if diag.placebo_status != "pass":
            warnings.append("placebo tests")
        st.warning(
            f"Diagnostic concern(s) in: {', '.join(warnings)}. "
            "These tests are often omitted from published papers but matter for validity."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Sensitivity heatmap (centred on their spec) ───────────────────────────────
st.markdown("---")
st.subheader("Sensitivity Heatmap — Centred on Reported Specification")
st.caption(
    "Color = RD estimate. The dashed vertical line marks the reported bandwidth. "
    "The shaded band shows the stable region. Instability near the reported spec is a red flag."
)
fig_heat = sensitivity_heatmap.plot_interactive(audit.bw_grid)

# Add a vertical line at the reported bandwidth
fig_heat.add_vline(
    x=reported_bw,
    line_dash="dot",
    line_color="#c0392b",
    line_width=2.5,
    annotation_text=f"Reported BW={reported_bw:.3f}",
    annotation_position="top left",
    annotation_font=dict(color="#c0392b", size=10),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Detailed diagnostic plots (expandable) ────────────────────────────────────
with st.expander("Density Test (McCrary)", expanded=(diag.mccrary_status != "pass")):
    fig_dens = density_plot.plot_interactive(diag.mccrary, cutoff=cutoff)
    st.plotly_chart(fig_dens, use_container_width=True)

with st.expander("Placebo Test Distribution", expanded=(diag.placebo_status != "pass")):
    fig_plac = placebo_plot.plot_interactive(diag.placebo)
    st.plotly_chart(fig_plac, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Audit Summary")

rows = [
    ("Estimate reproduction",
     f"{est.our_estimate:.4f} vs {est.reported_estimate:.4f}",
     f"{est.diff_in_se:.2f} SEs",
     badge(est_status_map[est.status], est.status.upper())),
    ("Bandwidth ratio (reported/MSE-optimal)",
     f"{bw.reported_bw:.4f} / {bw.mse_optimal_bw:.4f}",
     f"{bw.ratio:.2f}×",
     badge(bw_status_map[bw.status], bw.status.upper())),
    ("In stable region",
     f"[{sr[0]:.4f}, {sr[1]:.4f}]",
     "Yes" if bw.in_stable_region else "No",
     badge("pass" if bw.in_stable_region else "fail", "YES" if bw.in_stable_region else "NO")),
    ("Fragility score (nearby BWs)",
     f"{frag.n_stable}/{frag.n_nearby} agree",
     f"{frag.fragility_score:.0%}",
     badge(frag_status_map[frag.status], frag.status.upper())),
    ("McCrary density test",
     f"p = {diag.mccrary.p_value:.4f}",
     diag.mccrary.conclusion,
     badge(diag.mccrary_status, diag.mccrary_status.upper())),
    ("Placebo rate",
     f"{audit.diagnostics.placebo.n_significant_placebos} / {int((~np.isnan(diag.placebo.placebo_estimates)).sum())}",
     diag.placebo_msg,
     badge(diag.placebo_status, diag.placebo_status.upper())),
    ("Overall credibility score",
     f"Grade {audit.credibility_grade}",
     f"{audit.credibility_score:.0f}/100",
     badge(_VERDICT_STATUS[audit.verdict], audit.verdict)),
]

df_summary = pd.DataFrame(rows, columns=["Check", "Detail", "Value", "Status"])
# Render as HTML to preserve badge markup
html_rows = ""
for _, row in df_summary.iterrows():
    html_rows += (
        f"<tr><td style='padding:6px 12px;color:#1a1a2e'>{row['Check']}</td>"
        f"<td style='padding:6px 12px;color:#555'>{row['Detail']}</td>"
        f"<td style='padding:6px 12px;color:#1a1a2e;font-weight:600'>{row['Value']}</td>"
        f"<td style='padding:6px 12px'>{row['Status']}</td></tr>"
    )

table_html = f"""
<table style="width:100%;border-collapse:collapse;font-size:0.9rem">
  <thead>
    <tr style="border-bottom:2px solid #e0e0e8">
      <th style="text-align:left;padding:8px 12px;color:#888899;text-transform:uppercase;font-size:0.78rem;letter-spacing:0.05em">Check</th>
      <th style="text-align:left;padding:8px 12px;color:#888899;text-transform:uppercase;font-size:0.78rem;letter-spacing:0.05em">Detail</th>
      <th style="text-align:left;padding:8px 12px;color:#888899;text-transform:uppercase;font-size:0.78rem;letter-spacing:0.05em">Value</th>
      <th style="text-align:left;padding:8px 12px;color:#888899;text-transform:uppercase;font-size:0.78rem;letter-spacing:0.05em">Status</th>
    </tr>
  </thead>
  <tbody>{html_rows}</tbody>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)
