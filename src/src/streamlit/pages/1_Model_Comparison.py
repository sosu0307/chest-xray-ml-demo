import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Model Comparison", layout="wide")

st.markdown(
    """
<style>
.block-container{
    max-width: 1200px !important;
    padding-top: 1.0rem !important;
    padding-bottom: 0.8rem !important;
}
h1 { margin-bottom: 0.15rem !important; }
h2 { margin-top: 0.5rem !important; margin-bottom: 0.3rem !important; }
.element-container { margin-bottom: 0.35rem !important; }

.info{
    border:1px solid #7ec8f8;
    border-radius:10px;
    padding:8px 10px;
    background:#e8fbff;
    font-size:.94rem;
    line-height:1.35;
}

.card{
    border:1px solid #e5e7eb;
    border-radius:10px;
    padding:10px 12px;
    background:#ffffff;
}
.card-title{
    font-size:.84rem;
    color:#6b7280;
    margin-bottom:2px;
}
.card-value{
    font-size:1.35rem;
    font-weight:700;
    color:#111827;
    line-height:1.05;
}
.small-muted{
    color:#6b7280;
    font-size:.86rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Model Comparison — ProCNN vs ResNet50")
st.caption("Final operating point comparison for screening (threshold = 0.40)")

# =========================================================
# Navigation
# =========================================================
n1, n2 = st.columns(2)
with n1:
    if st.button("← Back to Presentation", use_container_width=True):
        st.switch_page("0_Deeplearning_Presentation.py")
with n2:
    if st.button("Next ▶ Training Curves", use_container_width=True):
        st.switch_page("pages/2_Training_Curves.py")

# =========================================================
# Fixed final metrics @ threshold 0.40
# (replace if needed with your exact final values)
# =========================================================
resnet = {
    "recall": 0.95,
    "precision": 1.00,
    "f1": 0.9744,
    "fn": 1,
    "fp": 0,
}
procnn = {
    "recall": 0.85,
    "precision": 1.00,
    "f1": 0.9189,
    "fn": 3,
    "fp": 0,
}

cm_resnet = np.array([[20, 0], [1, 19]])
cm_procnn = np.array([[20, 0], [3, 17]])

# =========================================================
# 1) Policy + Key numbers (no table noise)
# =========================================================
st.markdown(
    """
<div class="info">
<b>Screening policy:</b> prioritize high recall to reduce missed COVID cases (FN).  
<b>Operating point:</b> threshold fixed at <b>0.40</b>.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("## Key comparison at threshold 0.40")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">ResNet50 Recall</div>
  <div class="card-value">{resnet['recall']:.2f}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">ProCNN Recall</div>
  <div class="card-value">{procnn['recall']:.2f}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">Missed COVID (FN)</div>
  <div class="card-value">{resnet['fn']} vs {procnn['fn']}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">False Positives (FP)</div>
  <div class="card-value">{resnet['fp']} vs {procnn['fp']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div class='small-muted'>ResNet50 reduces missed COVID cases by <b>{procnn['fn'] - resnet['fn']}</b> at equal FP.</div>",
    unsafe_allow_html=True,
)


# =========================================================
# 2) Compact confusion matrices
def plot_cm_compact(cm: np.ndarray, title: str):
    # kleiner + robust gegen Abschneiden
    fig, ax = plt.subplots(figsize=(3.0, 2.7), dpi=130, constrained_layout=True)
    im = ax.imshow(cm, cmap="YlGnBu", interpolation="nearest", aspect="equal")

    ax.set_title(title, fontsize=10, pad=4, fontweight="bold")
    ax.set_xlabel("Pred", fontsize=8, labelpad=1)
    ax.set_ylabel("True", fontsize=8, labelpad=1)

    labels = ["Non-COVID", "COVID"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=7)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    # schlanke Colorbar, damit nichts rausläuft
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    return fig


# =========================================================

st.markdown("## Confusion matrices (compact)")
c1, c2 = st.columns(2)

with c1:
    fig1 = plot_cm_compact(cm_resnet, "ResNet50 @ 0.40")
    st.pyplot(fig1, use_container_width=True)  # wichtig
    plt.close(fig1)

with c2:
    fig2 = plot_cm_compact(cm_procnn, "ProCNN @ 0.40")
    st.pyplot(fig2, use_container_width=True)  # wichtig
    plt.close(fig2)

# =========================================================
# 3) Clear final recommendation
# =========================================================
st.markdown("## Recommendation")
st.markdown(
    """
- **Use ResNet50 @ threshold 0.40** as the primary screening model.  
- Reason: **fewer missed COVID cases (FN 1 vs 3)** with the **same FP (0)** in this evaluation.
"""
)
