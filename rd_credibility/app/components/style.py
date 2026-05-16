"""Custom CSS for the RD Credibility Dashboard."""

CUSTOM_CSS = """
<style>
/* ── Typography ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    color: #1a1a2e;
}

/* ── Page background ────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #f5f5f8;
    border-right: 1px solid #e0e0e8;
}
[data-testid="stSidebar"] h2 {
    color: #1a1a2e;
    font-size: 1.05rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-top: 1rem;
}

/* ── Page header ─────────────────────────────────────────────────── */
h1 {
    color: #1a1a2e;
    font-size: 1.7rem;
    font-weight: 700;
    border-bottom: 2px solid #e8e8e8;
    padding-bottom: 0.4rem;
    margin-bottom: 0.3rem;
}
h2 { color: #1a1a2e; font-size: 1.25rem; font-weight: 600; }
h3 { color: #1a1a2e; font-size: 1.05rem; font-weight: 600; }

/* ── Page description ────────────────────────────────────────────── */
.page-desc {
    color: #555566;
    font-size: 0.93rem;
    margin-bottom: 1.2rem;
    font-style: italic;
}

/* ── KPI cards ───────────────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e0e0e8;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(26,26,46,0.06);
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    color: #888899;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.25rem;
}

/* ── Status badges ────────────────────────────────────────────────── */
.badge-pass {
    display: inline-block;
    background: #e8f5e9;
    color: #27ae60;
    border: 1px solid #27ae60;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-warn {
    display: inline-block;
    background: #fff8e1;
    color: #f39c12;
    border: 1px solid #f39c12;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-fail {
    display: inline-block;
    background: #fdecea;
    color: #c0392b;
    border: 1px solid #c0392b;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ── Warning banner ───────────────────────────────────────────────── */
.warning-banner {
    background: #fff3cd;
    border-left: 4px solid #f39c12;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin: 0.8rem 0;
    color: #7d4b00;
    font-size: 0.9rem;
}
.error-banner {
    background: #fdecea;
    border-left: 4px solid #c0392b;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin: 0.8rem 0;
    color: #6b0000;
    font-size: 0.9rem;
}

/* ── Tables ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #e0e0e8 !important;
    border-radius: 6px;
}

/* ── Expander (learning mode) ─────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #c8d8e8;
    border-radius: 6px;
    background: #f0f6ff;
}

/* ── Remove Streamlit branding ────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_desc(text: str) -> None:
    import streamlit as st
    st.markdown(f'<p class="page-desc">{text}</p>', unsafe_allow_html=True)


def kpi_card(label: str, value: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f"</div>"
    )


def badge(status: str, text: str) -> str:
    cls = {"pass": "badge-pass", "warn": "badge-warn", "fail": "badge-fail"}.get(status, "badge-warn")
    return f'<span class="{cls}">{text}</span>'
