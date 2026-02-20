# pages/2_Training_Curves.py
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Training Curves", layout="wide")

# =========================================================
# Style (fixed + complete)
# =========================================================
st.markdown(
    """
<style>
/* ---------- Global layout ---------- */
.block-container{
    padding-top: 1.4rem !important;
    padding-bottom: 1.4rem !important;
    max-width: 1280px;
}
h1, .stMarkdown h1{
    margin-top: 0 !important;
    line-height: 1.2 !important;
}
h2, .stMarkdown h2, h3, .stMarkdown h3{
    line-height: 1.25 !important;
}

/* ---------- Cards ---------- */
.card{
    border: 1px solid #cfd8e3;
    border-radius: 12px;
    padding: 12px 14px;
    background: #ffffff;
}
.info-card{
    border: 1px solid #7ec8f8;
    border-radius: 12px;
    padding: 12px 14px;
    background: #e8fbff;
}

/* ---------- KPI cards (missing before) ---------- */
.kpi-card{
    border: 1px solid #9ed7fb;
    border-radius: 12px;
    padding: 12px 14px;
    background: #eaf6ff;
    min-height: 112px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-label{
    font-size: 0.95rem;
    color: #334155;
    margin-bottom: 4px;
    font-weight: 600;
}
.kpi-value{
    font-size: 2rem;
    line-height: 1.1;
    color: #0f172a;
    font-weight: 800;
    margin-bottom: 4px;
}
.small-muted{
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
}

/* ---------- Buttons ---------- */
div.stButton > button{
    border-radius: 10px !important;
}
.nav-space{
    margin-top: 0.2rem;
    margin-bottom: 0.55rem;
}

/* ---------- Plot wrapper ---------- */
.plot-card{
    border: 1px solid #d8e1ec;
    border-radius: 12px;
    background: #ffffff;
    padding: 8px 8px 0 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Header
# =========================================================
st.title("Training Curves")
st.caption("Show convergence + best-epoch evidence (compact presentation view)")

# =========================================================
# Navigation
# =========================================================
st.markdown('<div class="nav-space"></div>', unsafe_allow_html=True)
n1, n2, n3 = st.columns(3)
with n1:
    if st.button("← Back to Model Comparison", use_container_width=True):
        st.switch_page("pages/1_Model_Comparison.py")
with n2:
    if st.button("Back to Presentation", use_container_width=True):
        st.switch_page("0_Deeplearning_Presentation.py")
with n3:
    if st.button("Next ▶ Inference Demo", use_container_width=True):
        st.switch_page("pages/3_Inference_Demo.py")

# =========================================================
# Final ResNet run values (from CSV)
# =========================================================
epochs = list(range(1, 11))
train_loss = [
    0.2158805,
    0.0932772,
    0.0657117,
    0.0508374,
    0.0468090,
    0.0374746,
    0.0323460,
    0.0353059,
    0.0179156,
    0.0102567,
]
val_loss = [
    0.0937941,
    0.0405187,
    0.0694241,
    0.0694745,
    0.0472699,
    0.0672393,
    0.0373988,
    0.0680543,
    0.0251688,
    0.0282951,
]
val_f1 = [
    0.8912467,
    0.9564411,
    0.9120410,
    0.9267841,
    0.9592417,
    0.9368421,
    0.9576958,
    0.9534663,
    0.9833024,
    0.9822926,
]

best_f1 = max(val_f1)
best_epoch = val_f1.index(best_f1) + 1
last_val_loss = val_loss[-1]

# =========================================================
# KPI cards (now styled correctly)
# =========================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">ResNet50 epochs</div>
  <div class="kpi-value">{len(epochs)}</div>
  <div class="small-muted">final run</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Best val F1</div>
  <div class="kpi-value">{best_f1:.4f}</div>
  <div class="small-muted">held-out validation</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Best epoch</div>
  <div class="kpi-value">{best_epoch}</div>
  <div class="small-muted">checkpoint selected</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Last val loss</div>
  <div class="kpi-value">{last_val_loss:.4f}</div>
  <div class="small-muted">epoch 10</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("### ResNet50 Convergence (final run)")

# =========================================================
# Charts
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(6.6, 4.0))
    ax1.plot(epochs, train_loss, marker="o", label="train_loss")
    ax1.plot(epochs, val_loss, marker="o", label="val_loss")
    ax1.set_title("Loss vs Epoch (ResNet50)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.0))
    ax2.plot(epochs, val_f1, marker="o", label="val_f1")
    ax2.axvline(
        best_epoch, linestyle="--", linewidth=1.5, label=f"best epoch = {best_epoch}"
    )
    ax2.set_title("F1 vs Epoch (ResNet50)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("F1")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="info-card">
<b>Interpretation:</b> Convergence is stable; the best checkpoint is at <b>epoch {best_epoch}</b> with 
<b>val F1 = {best_f1:.4f}</b>. This supports selecting ResNet50 for deployment-style screening policy.
</div>
""",
    unsafe_allow_html=True,
)

st.caption("For educational and demonstration use only. Not for clinical diagnosis.")
