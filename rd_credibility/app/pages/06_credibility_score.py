"""Full credibility report page with PDF export."""

import io
import tempfile
import os
import numpy as np
import streamlit as st

st.set_page_config(page_title="Credibility Score · RD Dashboard", layout="wide")

from rd_credibility.app.components.style import inject_css, page_desc, badge, kpi_card
from rd_credibility.app.components.learning_mode import render_explainer
from rd_credibility.visualization import score_gauge, export

inject_css()

if "results" not in st.session_state:
    st.warning("Return to the main page to load data first.")
    st.stop()

results = st.session_state["results"]
cfg = st.session_state["cfg"]
y = st.session_state["y"]
x = st.session_state["x"]
report = results["report"]

st.title("RD Credibility Report")
page_desc(
    "Full credibility assessment: component scores, explanations, recommendations, "
    "and a publication-quality PDF export."
)

# ── Grade banner ──────────────────────────────────────────────────────────────
grade_status = {"A": "pass", "B": "pass", "C": "warn", "D": "fail", "F": "fail"}
status = grade_status.get(report.grade, "warn")
st.markdown(
    badge(status, f"Grade {report.grade}  ·  {report.total_score:.0f}/100"),
    unsafe_allow_html=True,
)
st.markdown(f"> {report.summary}")
st.markdown("---")

# ── Gauge ─────────────────────────────────────────────────────────────────────
st.subheader("Score Gauge")
fig_gauge = score_gauge.plot_interactive(report)
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ── Component breakdown ────────────────────────────────────────────────────────
st.subheader("Component Scores")

_ICONS = {"manipulation": "🔬", "balance": "⚖️", "sensitivity": "📊", "placebo": "🎭"}
_NAMES = {
    "manipulation": "Manipulation (McCrary)",
    "balance": "Covariate Balance",
    "sensitivity": "Bandwidth Sensitivity",
    "placebo": "Placebo Tests",
}

cols = st.columns(4)
for col, key in zip(cols, ["manipulation", "balance", "sensitivity", "placebo"]):
    score = report.component_scores.get(key, 0)
    scaled = score * 4  # 0-25 → 0-100
    s = "pass" if scaled >= 70 else ("warn" if scaled >= 40 else "fail")
    col.markdown(kpi_card(f"{_ICONS[key]} {_NAMES[key]}", f"{score:.0f}/25"), unsafe_allow_html=True)

st.markdown("")

for key in ["manipulation", "balance", "sensitivity", "placebo"]:
    score = report.component_scores.get(key, 0)
    expl = report.component_explanations.get(key, "")
    with st.expander(
        f"{_ICONS[key]} {_NAMES[key]} — {score:.0f}/25",
        expanded=(score < 15),
    ):
        st.markdown(f"**Score:** {score:.0f} / 25")
        st.markdown(f"**Assessment:** {expl}")

        # Pull relevant diagnostic result
        diag_key = {
            "manipulation": "mccrary",
            "balance": "balance",
            "sensitivity": "bw_grid",
            "placebo": "placebo",
        }[key]
        lm_key = {
            "manipulation": "mccrary",
            "balance": "balance",
            "sensitivity": "sensitivity",
            "placebo": "placebo",
        }[key]

        if cfg.get("learning_mode"):
            render_explainer(lm_key, results.get(diag_key))

st.markdown("---")

# ── Warnings and recommendations ──────────────────────────────────────────────
if report.warnings:
    st.subheader("Warnings")
    for w in report.warnings:
        st.markdown(f'<div class="error-banner">⚠️ {w}</div>', unsafe_allow_html=True)

if report.recommendations:
    st.subheader("Recommendations")
    for i, rec in enumerate(report.recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

st.markdown("---")

# ── PDF export ─────────────────────────────────────────────────────────────────
st.subheader("Export Report")
st.caption(
    "Generates a publication-quality 2×2 panel figure (RD plot, density, "
    "covariate balance, placebo) saved as PDF and PNG."
)

if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Generating report…"):
        all_results = {
            "y": y,
            "x": x,
            "cutoff": cfg["cutoff"],
            "bandwidth": results["bandwidth_used"],
            "poly_order": cfg["poly_order"],
            "mccrary": results["mccrary"],
            "balance": results["balance"],
            "placebo": results["placebo"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "rd_credibility_panel")
            fig = export.save_publication_panel(all_results, output_path=out_path)

            pdf_path = out_path + ".pdf"
            png_path = out_path + ".png"

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            with open(png_path, "rb") as f:
                png_bytes = f.read()

        import matplotlib.pyplot as plt
        plt.close(fig)

    dataset_slug = (cfg.get("dataset_name") or "rd").replace(" ", "_").replace("(", "").replace(")", "")

    col_pdf, col_png = st.columns(2)
    col_pdf.download_button(
        label="⬇ Download PDF",
        data=pdf_bytes,
        file_name=f"{dataset_slug}_rd_panel.pdf",
        mime="application/pdf",
    )
    col_png.download_button(
        label="⬇ Download PNG",
        data=png_bytes,
        file_name=f"{dataset_slug}_rd_panel.png",
        mime="image/png",
    )

    st.success("Report generated. Click the buttons above to download.")
    st.image(png_bytes, caption="Publication panel preview", use_column_width=True)
