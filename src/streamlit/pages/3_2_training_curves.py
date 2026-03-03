import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# =========================================================
# Paths (robust if script is inside src/streamlit/pages/)
# =========================================================
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "pages":
    STREAMLIT_DIR = _THIS_FILE.parent.parent  # src/streamlit
else:
    STREAMLIT_DIR = _THIS_FILE.parent

# =========================================================
# FIXED PAGE PATHS (adjust if needed)
# =========================================================
PAGE_MODEL_COMPARISON = "pages/3_1_model_comparison.py"
PAGE_INFERENCE_DEMO = "pages/3_3_inference_demo.py"

# =========================================================
# STYLE (consistent with 3.0 / 3.1 pages)
# =========================================================
st.markdown(
    """
<style>
.block-container{
    padding-top: 1.0rem;
    padding-bottom: 1.2rem;
    max-width: 1180px;
}
h1, h2, h3 { margin-bottom: 0.25rem; }
hr { margin-top: 0.7rem !important; margin-bottom: 0.8rem !important; }

.small-muted{
    color:#5f6b7a;
    font-size:0.93rem;
}

/* Cards */
.card, .info-card, .success-card, .warn-card, .soft-card{
    border-radius:14px;
    padding:14px 16px;
    margin-bottom:12px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.03);
}
.card{
    border:1px solid #d9e1ea;
    background:#ffffff;
}
.info-card{
    border:1px solid #bfe3ff;
    background:#eef7ff;
}
.success-card{
    border:1px solid #b7e3cc;
    background:#eefaf4;
}
.warn-card{
    border:1px solid #ffd27d;
    background:#fff8e8;
}
.soft-card{
    border:1px solid #dfe7ef;
    background:#fbfdff;
}

/* KPI cards */
.kpi-card{
    border:1px solid #cfe7ff;
    border-radius:14px;
    padding:12px 10px;
    background:#f6fbff;
    text-align:center;
    height:100%;
}
.kpi-title{
    color:#5f6b7a;
    font-size:0.80rem;
    margin-bottom:4px;
}
.kpi-value{
    font-size:1.08rem;
    font-weight:700;
    line-height:1.15;
}
.kpi-sub{
    color:#5f6b7a;
    font-size:0.82rem;
    margin-top:4px;
}

/* Plot wrapper */
.plot-card{
    border:1px solid #d8e1ec;
    border-radius:12px;
    background:#ffffff;
    padding:10px 10px 4px 10px;
    margin-bottom:12px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.03);
}

/* Buttons */
div.stButton > button{
    min-height: 54px !important;
    border-radius: 12px !important;
    font-size: 0.98rem !important;
    font-weight: 700 !important;
}
div.stButton > button p{
    font-size: 0.98rem !important;
    font-weight: 700 !important;
}

/* Sidebar readability */
[data-testid="stSidebarNav"] *{
    font-size: 0.98rem !important;
}
[data-testid="stSidebarNav"] a{
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .info-card ul, .success-card ul, .warn-card ul, .card ul{
        margin: 0;
        padding-left: 1.05rem;
    }
    .info-card li, .success-card li, .warn-card li, .card li{
        margin-bottom: 0.28rem;
    }
    .info-card li:last-child, .success-card li:last-child, .warn-card li:last-child, .card li:last-child{
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================
def card(html: str, cls: str = "card"):
    st.markdown(f"<div class='{cls}'>{html}</div>", unsafe_allow_html=True)


def kpi(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_switch(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        st.warning(
            f"Could not open page: `{page_path}`. Check file name/path in `src/streamlit/pages/`."
        )


def nav_row(top: bool = True):
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "⬅ Back: Model Comparison",
            use_container_width=True,
            key=f"{'top' if top else 'bottom'}_back_model",
        ):
            safe_switch(PAGE_MODEL_COMPARISON)
    with c2:
        if st.button(
            "Next: Inference Demo ➜",
            use_container_width=True,
            key=f"{'top' if top else 'bottom'}_next_inference",
        ):
            safe_switch(PAGE_INFERENCE_DEMO)


# =========================================================
# Embedded training values (ResNet50 final run)
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

final_epoch = epochs[-1]
final_val_f1 = val_f1[-1]
final_val_loss = val_loss[-1]
final_train_loss = train_loss[-1]

delta_best_vs_final_f1 = best_f1 - final_val_f1
best_val_loss = val_loss[best_epoch - 1]
best_train_loss = train_loss[best_epoch - 1]

# =========================================================
# Header + navigation
# =========================================================
st.title("Training Curves — ResNet50 Final Run")
st.caption(
    "Convergence view for the selected final model. This page provides visual evidence for "
    "checkpoint selection using validation F1 across epochs."
)

nav_row(top=True)
st.markdown("---")

# =========================================================
# Page scope (compact)
# =========================================================
card(
    """
    <h4 style="margin:0 0 8px 0;">Fair-Comparison Note</h4>
    <ul style="margin:0; padding-left: 1.05rem;">
      <li>All models use the <b>same held-out test set (n = 3175)</b>.</li>
      <li>Metrics and <b>TN / FP / FN / TP</b> are directly comparable.</li>
      <li>For screening decisions, prioritize <b>Recall</b> and <b>FN</b>.</li>
    </ul>
    """,
    cls="info-card",
)

# =========================================================
# KPI snapshot
# =========================================================
st.subheader("Training Snapshot (ResNet50 final run)")

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi("Epochs", f"{len(epochs)}", "final run")
with k2:
    kpi("Best Val F1", f"{best_f1:.4f}", "validation")
with k3:
    kpi("Best Epoch", f"{best_epoch}", "checkpoint selected")
with k4:
    kpi("Final Val F1", f"{final_val_f1:.4f}", f"epoch {final_epoch}")
with k5:
    kpi("Last Val Loss", f"{final_val_loss:.4f}", f"epoch {final_epoch}")
with k6:
    kpi("Δ Best vs Final F1", f"{delta_best_vs_final_f1:.4f}", "best - final")

st.markdown("---")

# =========================================================
# Charts
# =========================================================
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.plot(epochs, train_loss, marker="o", label="train_loss")
    ax1.plot(epochs, val_loss, marker="o", label="val_loss")
    ax1.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.3,
        label=f"selected checkpoint = epoch {best_epoch}",
    )
    ax1.set_title("Loss vs Epoch (ResNet50)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
    ax2.plot(epochs, val_f1, marker="o", label="val_f1")
    ax2.axvline(
        best_epoch, linestyle="--", linewidth=1.5, label=f"best epoch = {best_epoch}"
    )
    ax2.scatter([best_epoch], [best_f1], s=80, zorder=3)
    ax2.annotate(
        f"Highest F1 at epoch {best_epoch}\n({best_f1:.4f})",
        xy=(best_epoch, best_f1),
        xytext=(best_epoch - 2.2, best_f1 - 0.030),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=9,
    )
    ax2.set_title("Validation F1 vs Epoch (ResNet50)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Professional takeaway box (replaces "Interpretation")
# =========================================================
card(
    f"""
    <h4 style="margin:0 0 8px 0;">Key Takeaways (checkpoint selection)</h4>
    <ul style="margin:0; padding-left:1.05rem;">
      <li><b>Convergence is stable</b>: train loss trends downward and validation performance remains high in late epochs.</li>
      <li><b>Best validation F1 occurs at epoch {best_epoch}</b> (<b>{best_f1:.4f}</b>), which is why this checkpoint is selected for downstream evaluation.</li>
      <li><b>Epoch {final_epoch} remains very close</b> (<b>Val F1 = {final_val_f1:.4f}</b>), but checkpoint selection follows the highest validation F1 criterion.</li>
    </ul>
    """,
    cls="success-card",
)

card(
    f"""
    <b>Decision note:</b> The selected checkpoint is <b>epoch {best_epoch}</b> (not simply the last epoch),
    which reflects a validation-driven model selection policy and supports reproducible downstream results
    in the inference demo and reliability checks.
    """,
    cls="soft-card",
)

# =========================================================
# Bottom navigation
# =========================================================
st.markdown("---")
nav_row(top=False)
