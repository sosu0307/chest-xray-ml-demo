from pathlib import Path
import streamlit as st

# Nur hier page config setzen
st.set_page_config(page_title="ML Project Presentation")

st.markdown(
    """
<style>
/* Make main content truly wide */
.block-container {
  max-width: 100% !important;
  padding-left: 2rem;
  padding-right: 2rem;
  padding-top: 1.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent  # src/streamlit

# Custom sidebar titles (ohne Zahlen)
pg = st.navigation(
    [
        st.Page(str(APP_DIR / "0_overview.py"), title="Overview"),
        st.Page(
            str(APP_DIR / "pages" / "1_eda_feature_engineering.py"),
            title="EDA Feature Engineering",
        ),
        st.Page(str(APP_DIR / "pages" / "2_classical_ml.py"), title="Classical ML"),
        st.Page(str(APP_DIR / "pages" / "3_0_deep_learning.py"), title="Deep Learning"),
        st.Page(
            str(APP_DIR / "pages" / "3_1_model_comparison.py"), title="Model Comparison"
        ),
        st.Page(
            str(APP_DIR / "pages" / "3_2_training_curves.py"), title="Training Curves"
        ),
        st.Page(
            str(APP_DIR / "pages" / "3_3_inference_demo.py"), title="Inference Demo"
        ),
        st.Page(str(APP_DIR / "pages" / "3_4_gradcam_demo.py"), title="Grad-CAM Demo"),
        st.Page(
            str(APP_DIR / "pages" / "4_conclusion_questions.py"),
            title="Conclusion & Questions",
        ),
    ]
)

pg.run()
