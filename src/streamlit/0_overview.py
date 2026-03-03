import streamlit as st

# -----------------------------
# Style (clean / professional)
# -----------------------------
st.markdown(
    """
<style>
/* Page width + spacing */
.main .block-container{
    max-width: 980px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Typography */
h1, h2, h3, h4 {
    letter-spacing: -0.2px;
}
.small-muted {
    color: #667085;
    font-size: 0.95rem;
    line-height: 1.45;
}

/* Hero */
.hero {
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    padding: 22px 22px 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.hero-title {
    font-size: 2.05rem;
    font-weight: 700;
    color: #1f2937;
    margin: 0 0 6px 0;
}
.hero-subtitle {
    color: #667085;
    font-size: 0.98rem;
    margin: 0;
}

/* Small chips */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}
.chip {
    display: inline-block;
    border: 1px solid #d0d5dd;
    background: #f9fafb;
    color: #344054;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.82rem;
}

/* Info card */
.info-card {
    border: 1px solid #cfe3ff;
    border-radius: 16px;
    padding: 16px 18px;
    background: #f4f8ff;
    margin-bottom: 18px;
}
.info-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: #1d2939;
    margin-bottom: 6px;
}
.info-text {
    margin: 0;
    color: #344054;
    line-height: 1.55;
}

/* Section cards (stacked) */
.section-card {
    border: 1px solid #e4e7ec;
    border-radius: 16px;
    background: #ffffff;
    padding: 16px 18px 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.section-card.success {
    border-color: #b7e3cc;
    background: #f2fbf6;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.section-number {
    width: 32px;
    height: 32px;
    min-width: 32px;
    border-radius: 50%;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #334155;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.95rem;
}
.section-card.success .section-number {
    border-color: #9fd7bb;
    background: #eafaf1;
    color: #146c43;
}
.section-title {
    font-size: 1.22rem;
    font-weight: 700;
    color: #1f2937;
    margin: 0;
}
.section-sub {
    color: #667085;
    font-size: 0.9rem;
    margin: 0;
}

.clean-list {
    margin: 8px 0 0 0;
    padding-left: 1.15rem;
    color: #344054;
    line-height: 1.6;
}
.clean-list li {
    margin-bottom: 2px;
}

/* Divider title spacing */
.section-heading {
    margin-top: 0.5rem;
    margin-bottom: 0.4rem;
}

/* Buttons */
div.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #d0d5dd !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    font-weight: 600 !important;
    height: 42px !important;
    box-shadow: 0 1px 2px rgba(16,24,40,0.03);
}
div.stButton > button:hover {
    border-color: #98a2b3 !important;
    background: #f9fafb !important;
}

/* Button wrappers (to visually group button with card) */
.button-wrap {
    margin-top: -2px;
    margin-bottom: 14px;
}

/* Footer note */
.footer-note {
    color: #667085;
    font-size: 0.88rem;
    margin-top: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header / Hero
# -----------------------------
st.markdown(
    """
<div class="hero">
    <div class="hero-title">COVID-19 Chest X-Ray ML Project</div>
    <p class="hero-subtitle">
        End-to-end workflow from EDA to model selection, threshold tuning, and reliability checks
    </p>
    <div class="chip-row">
        <span class="chip">EDA</span>
        <span class="chip">Classical ML</span>
        <span class="chip">Deep Learning</span>
        <span class="chip">ResNet50 Focus</span>
        <span class="chip">Grad-CAM</span>
        <span class="chip">Threshold Tuning</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Project Goal
# -----------------------------
st.markdown(
    """
<div class="info-card">
    <div class="info-title">Project Goal</div>
    <p class="info-text">
        Build and compare machine learning approaches for COVID-19 screening from chest X-ray images.
        We evaluate the full pipeline from data understanding (EDA) to final model selection and
        reliability checks to support a transparent decision.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Story Flow (stacked, not side-by-side)
# -----------------------------
st.markdown(
    "<h3 class='section-heading'>Presentation Flow</h3>", unsafe_allow_html=True
)
st.markdown(
    "<div class='small-muted' style='margin-bottom: 12px;'>A clear presentation story: data → baselines → deep learning → final decision.</div>",
    unsafe_allow_html=True,
)

# 1. EDA
st.markdown(
    """
<div class="section-card">
    <div class="section-header">
        <div class="section-number">1</div>
        <div>
            <p class="section-title">EDA & Feature Engineering</p>
            <p class="section-sub">Understand the dataset before modeling</p>
        </div>
    </div>
    <ul class="clean-list">
        <li>Dataset overview and class balance</li>
        <li>Example images and preprocessing pipeline</li>
        <li>Feature engineering choices, assumptions, and risks</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)
if st.button(
    "Open 1. EDA & Feature Engineering", use_container_width=True, key="go_eda"
):
    st.switch_page("pages/1_eda_feature_engineering.py")

# 2. Classical ML
st.markdown(
    """
<div class="section-card">
    <div class="section-header">
        <div class="section-number">2</div>
        <div>
            <p class="section-title">Classical Machine Learning</p>
            <p class="section-sub">Establish baselines and compare strengths/limitations</p>
        </div>
    </div>
    <ul class="clean-list">
        <li>Baseline models and evaluation metrics</li>
        <li>Strengths and limitations vs. deep learning</li>
        <li>Performance comparison to support the final choice</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)
if st.button("Open 2. Classical ML", use_container_width=True, key="go_classical"):
    st.switch_page("pages/2_classical_ml.py")

# 3. Deep Learning
st.markdown(
    """
<div class="section-card">
    <div class="section-header">
        <div class="section-number">3</div>
        <div>
            <p class="section-title">Deep Learning</p>
            <p class="section-sub">Model comparison, threshold tuning, and live demo</p>
        </div>
    </div>
    <ul class="clean-list">
        <li>Model comparison (focus: ResNet50)</li>
        <li>Training curves and threshold tuning</li>
        <li>Inference demo + Grad-CAM reliability checks</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)
if st.button("Open 3. Deep Learning", use_container_width=True, key="go_dl"):
    st.switch_page("pages/3_0_deep_learning.py")

# 4. Conclusion
st.markdown(
    """
<div class="section-card success">
    <div class="section-header">
        <div class="section-number">4</div>
        <div>
            <p class="section-title">Conclusion & Questions</p>
            <p class="section-sub">Summarize the decision and discuss next steps</p>
        </div>
    </div>
    <ul class="clean-list">
        <li>Final model selection and rationale</li>
        <li>Key takeaways, limitations, and risks</li>
        <li>Next steps and Q&amp;A</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)
if st.button(
    "Open 4. Conclusion & Questions", use_container_width=True, key="go_conclusion"
):
    st.switch_page("pages/4_conclusion_questions.py")

st.markdown("---")
st.markdown(
    "<div class='footer-note'>Tip: The sidebar also reflects the page order automatically based on file names (e.g., 0_, 1_, 2_, ...).</div>",
    unsafe_allow_html=True,
)
