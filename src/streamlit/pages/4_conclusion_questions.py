# pages/4_conclusion_questions.py
from pathlib import Path
import streamlit as st

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Conclusion & Questions", layout="wide")

# =========================================================
# Paths (robust for src/streamlit/pages/)
# =========================================================
CURRENT_FILE = Path(__file__).resolve()

if CURRENT_FILE.parent.name == "pages":
    STREAMLIT_DIR = CURRENT_FILE.parent.parent  # src/streamlit
else:
    STREAMLIT_DIR = CURRENT_FILE.parent  # src/streamlit

PAGE_GRADCAM_DEMO = "pages/3_4_gradcam_demo.py"

# =========================================================
# Style (cleaner / less text-heavy)
# =========================================================
st.markdown(
    """
<style>
.block-container{
    padding-top: 1.05rem !important;
    padding-bottom: 1.25rem !important;
    max-width: 1080px;
}

h1, .stMarkdown h1{
    margin-top: 0 !important;
    line-height: 1.12 !important;
}
h2, .stMarkdown h2, h3, .stMarkdown h3{
    line-height: 1.18 !important;
    margin-top: 0.12rem !important;
}

.small-muted{
    color:#6b7280;
    font-size:0.94rem;
}

/* Cards */
.card{
    border:1px solid #d9e1ea;
    border-radius:14px;
    padding:14px 16px;
    background:#ffffff;
    margin-bottom:12px;
}
.info-card{
    border:1px solid #b9dcff;
    border-radius:14px;
    padding:14px 16px;
    background:#eef7ff;
    margin-bottom:12px;
}
.success-card{
    border:1px solid #b7e3cc;
    border-radius:14px;
    padding:14px 16px;
    background:#eefaf4;
    margin-bottom:12px;
}
.warn-card{
    border:1px solid #f2c46c;
    border-radius:14px;
    padding:14px 16px;
    background:#fff8e8;
    margin-bottom:12px;
}

/* Compact KPI pills */
.kpi-pill{
    border:1px solid #cfe7ff;
    border-radius:14px;
    background:#f6fbff;
    padding:10px 12px;
    text-align:center;
    min-height:72px;
}
.kpi-pill .label{
    color:#5f6b7a;
    font-size:0.80rem;
    margin-bottom:4px;
}
.kpi-pill .value{
    font-size:1.02rem;
    font-weight:700;
    line-height:1.15;
    color:#1f2937;
}

/* Buttons */
div.stButton > button{
    min-height: 50px !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
div.stButton > button p{
    font-weight: 700 !important;
}

/* Tabs spacing */
.stTabs [data-baseweb="tab-list"]{
    gap: 0.75rem;
}
.stTabs [data-baseweb="tab"]{
    padding-left: 0.25rem;
    padding-right: 0.25rem;
}

hr{
    margin-top: 0.8rem !important;
    margin-bottom: 0.8rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================
def safe_switch(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        st.warning(f"Could not open page: {page_path}")


def card(html: str, cls: str = "card"):
    st.markdown(f"<div class='{cls}'>{html}</div>", unsafe_allow_html=True)


def pill(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-pill">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Header
# =========================================================
st.title("Conclusion & Questions")
st.markdown(
    "<div class='small-muted'>COVID vs Non-COVID chest X-ray classification — final project summary</div>",
    unsafe_allow_html=True,
)

# =========================================================
# Top navigation
# =========================================================
btn_col, _ = st.columns([1.15, 2.2], gap="large")
with btn_col:
    if st.button(
        "← Back to Grad-CAM Demo", use_container_width=True, key="back_gradcam_top"
    ):
        safe_switch(PAGE_GRADCAM_DEMO)

st.markdown("---")

# =========================================================
# Compact summary row (minimal text, easy to present)
# =========================================================
p1, p2, p3, p4 = st.columns(4)
with p1:
    pill("Final Model", "ResNet50")
with p2:
    pill("Decision Policy", "Recall-first")
with p3:
    pill("Use Case", "Screening Support")
with p4:
    pill("Status", "Project Prototype")

# =========================================================
# Single final takeaway (merged; removes duplication)
# =========================================================
st.subheader("Final Takeaway")
card(
    """
    <ul style="margin:0; padding-left: 1.05rem;">
      <li><b>ResNet50</b> was selected for the final demo pipeline (recall-first screening objective).</li>
      <li><b>ResNet50 outputs probabilities</b>.</li>
      <li>The <b>final prediction depends on the decision threshold</b>.</li>
      <li>We use a <b>recall-first threshold</b> to reduce missed COVID cases.</li>
      <li><b>Grad-CAM</b> is a supportive interpretability check, not clinical proof.</li>
    </ul>
    """,
    cls="info-card",
)

# =========================================================
# Scope / limitations / next steps (short + tabbed)
# =========================================================
st.subheader("Scope, Limitations & Next Steps")

tab_scope, tab_limits, tab_next = st.tabs(
    ["Scope", "Limitations", "Next Practical Steps"]
)

with tab_scope:
    card(
        """
        <ul style="margin:0; padding-left: 1.05rem;">
          <li>This project demonstrates an <b>end-to-end ML workflow</b> (training, evaluation, thresholding, demo, interpretability).</li>
          <li>It is presented as an <b>educational decision-support prototype</b> for portfolio / project demonstration.</li>
        </ul>
        """,
        cls="card",
    )

with tab_limits:
    card(
        """
        <ul style="margin:0; padding-left: 1.05rem;">
          <li><b>Not</b> a clinically validated diagnostic system.</li>
          <li>Results depend on the <b>dataset</b>, preprocessing choices, and evaluation setup used in this project.</li>
        </ul>
        """,
        cls="warn-card",
    )

with tab_next:
    card(
        """
        <ul style="margin:0; padding-left: 1.05rem;">
          <li>Run <b>external validation</b> on an independent dataset / site.</li>
          <li>Perform <b>threshold calibration</b> for the target operational recall/FP trade-off.</li>
          <li>Package the app with <b>pinned dependencies</b> for reproducible deployment.</li>
        </ul>
        """,
        cls="success-card",
    )

# =========================================================
# Closing box
# =========================================================
st.markdown("---")
card(
    """
    <h3 style="margin:0;">Thank you — Questions?</h3>
    """,
    cls="info-card",
)

# =========================================================
# Bottom navigation
# =========================================================
bcol, _ = st.columns([1.15, 2.2], gap="large")
with bcol:
    if st.button(
        "← Back to Grad-CAM Demo", use_container_width=True, key="back_gradcam_bottom"
    ):
        safe_switch(PAGE_GRADCAM_DEMO)
