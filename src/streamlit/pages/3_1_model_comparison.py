import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.title("Model Comparison")

# =========================================================
# Paths (robust if script is inside src/streamlit/pages/)
# =========================================================
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "pages":
    STREAMLIT_DIR = _THIS_FILE.parent.parent  # src/streamlit
else:
    STREAMLIT_DIR = _THIS_FILE.parent

# =========================================================
# Navigation targets
# =========================================================
PAGE_DEEP_LEARNING_OVERVIEW = "pages/3_deep_learning.py"  # adjust if needed
PAGE_TRAINING_CURVES = "pages/3_2_training_curves.py"  # adjust if needed

# =========================================================
# Style
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
.card, .info-card, .success-card, .soft-card{
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
.soft-card{
    border:1px solid #dfe7ef;
    background:#fbfdff;
}

/* KPI cards */
.kpi-card{
    border:1px solid #dfe7ef;
    border-radius:14px;
    padding:12px 10px;
    background:#fbfdff;
    text-align:center;
    height:100%;
}
.kpi-good{
    border:1px solid #b7e3cc;
    background:#eefaf4;
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
    margin-top:3px;
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


# =========================================================
# Helpers
# =========================================================
CM_CMAP = "Blues"  # Alternativen: "BuGn", "cividis", "Purples"


def card(html: str, cls: str = "card"):
    st.markdown(f"<div class='{cls}'>{html}</div>", unsafe_allow_html=True)


def kpi(title: str, value: str, sub: str = "", good: bool = False):
    cls = "kpi-card kpi-good" if good else "kpi-card"
    st.markdown(
        f"""
        <div class="{cls}">
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
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(
            "⬅ Back to Deep Learning",
            use_container_width=True,
            key=f"{'top' if top else 'bottom'}_back",
        ):
            safe_switch(PAGE_DEEP_LEARNING_OVERVIEW)
    with c2:
        if st.button(
            "Next: Training Curves ➜",
            use_container_width=True,
            key=f"{'top' if top else 'bottom'}_next",
        ):
            safe_switch(PAGE_TRAINING_CURVES)


def plot_confusion_matrix(
    ax, cm: np.ndarray, title: str, vmax: int, cmap: str = CM_CMAP
):
    im = ax.imshow(cm, vmin=0, vmax=vmax, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([0, 1], labels=["Non-COVID", "COVID"])
    ax.set_yticks([0, 1], labels=["Non-COVID", "COVID"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    thresh = vmax * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "#111111"
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )
    return im


# =========================================================
# Static Report-2 comparison data (same held-out test set, threshold 0.40)
# =========================================================
N_TEST = 3175
THRESHOLD = 0.40

comparison_df = pd.DataFrame(
    [
        {
            "Model": "ProCNN (masked)",
            "Threshold": 0.40,
            "Accuracy": 0.9609,
            "Precision (COVID)": 0.8654,
            "Recall (COVID)": 0.9133,
            "F1 (COVID)": 0.8887,
            "TN": 2556,
            "FP": 77,
            "FN": 47,
            "TP": 495,
            "N": 3175,
        },
        {
            "Model": "ResNet50",
            "Threshold": 0.40,
            "Accuracy": 0.9918,
            "Precision (COVID)": 0.9725,
            "Recall (COVID)": 0.9797,
            "F1 (COVID)": 0.9761,
            "TN": 2618,
            "FP": 15,
            "FN": 11,
            "TP": 531,
            "N": 3175,
        },
    ]
)

# Compute deltas (ResNet50 - ProCNN)
procnn_row = comparison_df.loc[comparison_df["Model"] == "ProCNN (masked)"].iloc[0]
resnet_row = comparison_df.loc[comparison_df["Model"] == "ResNet50"].iloc[0]

delta_f1 = float(resnet_row["F1 (COVID)"] - procnn_row["F1 (COVID)"])
delta_fn = int(resnet_row["FN"] - procnn_row["FN"])  # negative is good
delta_fp = int(resnet_row["FP"] - procnn_row["FP"])  # negative is good
delta_recall = float(resnet_row["Recall (COVID)"] - procnn_row["Recall (COVID)"])
delta_precision = float(
    resnet_row["Precision (COVID)"] - procnn_row["Precision (COVID)"]
)

# Confusion matrices
cm_procnn = np.array([[2556, 77], [47, 495]])

cm_resnet50 = np.array([[2618, 15], [11, 531]])

# =========================================================
# Header + navigation
# =========================================================
st.title("Model Comparison — ProCNN (masked) vs ResNet50")
st.caption(
    f"Static comparison from Report 2 on the same held-out test set (n = {N_TEST}), threshold = {THRESHOLD:.2f}"
)

nav_row(top=True)
st.markdown("---")

# =========================================================
# Compact context note
# =========================================================
card(
    """
    <h4 style="margin:0 0 8px 0;">Page scope</h4>
    <ul style="margin:0; padding-left:1.05rem;">
      <li><b>Same held-out test set</b> for both models</li>
      <li><b>Same threshold (0.40)</b> for direct comparison</li>
      <li>Focus on <b>F1</b> and error counts (<b>FN / FP</b>) under a recall-first screening policy</li>
    </ul>
    """,
    cls="info-card",
)

# =========================================================
# KPI delta strip (presentation-friendly)
# =========================================================
st.subheader("Decision-Relevant Deltas (ResNet50 − ProCNN)")

d1, d2, d3, d4 = st.columns(4)
with d1:
    kpi("Δ COVID F1", f"{delta_f1:+.4f}", "higher is better", good=True)
with d2:
    kpi("Δ Recall (COVID)", f"{delta_recall:+.4f}", "higher is better", good=True)
with d3:
    # negative delta FN is GOOD (fewer FN)
    kpi("Δ FN", f"{delta_fn:+d}", "negative = fewer misses", good=True)
with d4:
    # negative delta FP is GOOD (fewer FP)
    kpi("Δ FP", f"{delta_fp:+d}", "negative = fewer false alarms", good=True)

st.markdown("---")

# =========================================================
# Tabs to reduce scroll and keep presentation calm
# =========================================================
tab_metrics, tab_cm = st.tabs(["Metric Table + Key Takeaways", "Confusion Matrices"])

# ---------------------------------------------------------
# Tab 1: Metrics + Takeaways
# ---------------------------------------------------------
with tab_metrics:
    st.subheader(f"Compact Metric Table (same test set, n = {N_TEST})")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Threshold": st.column_config.NumberColumn(format="%.1f"),
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Precision (COVID)": st.column_config.NumberColumn(format="%.4f"),
            "Recall (COVID)": st.column_config.NumberColumn(format="%.4f"),
            "F1 (COVID)": st.column_config.NumberColumn(format="%.4f"),
            "TN": st.column_config.NumberColumn(format="%d"),
            "FP": st.column_config.NumberColumn(format="%d"),
            "FN": st.column_config.NumberColumn(format="%d"),
            "TP": st.column_config.NumberColumn(format="%d"),
            "N": st.column_config.NumberColumn(format="%d"),
        },
    )

    # Show integer formatting for FN/FP delta rows visually by replacing display if desired
    card(
        f"""
        <h4 style="margin:0 0 8px 0;">Key Takeaways (Recall-first screening policy)</h4>
        <ul style="margin:0; padding-left:1.05rem;">
          <li><b>ResNet50 improves COVID F1</b> from <b>{procnn_row["F1 (COVID)"]:.4f}</b> to <b>{resnet_row["F1 (COVID)"]:.4f}</b> (<b>Δ = {delta_f1:+.4f}</b>).</li>
          <li><b>False negatives drop from {int(procnn_row["FN"])} to {int(resnet_row["FN"])}</b> (<b>Δ = {delta_fn:+d}</b>), which is the key improvement for a recall-first screening use case.</li>
          <li><b>False positives also drop from {int(procnn_row["FP"])} to {int(resnet_row["FP"])}</b> (<b>Δ = {delta_fp:+d}</b>), improving the overall error profile.</li>
        </ul>
        """,
        cls="success-card",
    )

# ---------------------------------------------------------
# Tab 2: Confusion Matrices
# ---------------------------------------------------------
with tab_cm:
    st.subheader(f"Confusion Matrices (Report values, threshold = {THRESHOLD:.2f})")

    note1, note2 = st.columns(2)
    with note1:
        card(
            """
            <b>ProCNN (masked) — Test set</b><br>
            Higher FN and FP than ResNet50 in this comparison.
            """,
            cls="soft-card",
        )
    with note2:
        card(
            """
            <b>ResNet50 — Test set</b><br>
            Lower FN and FP, supporting the recall-first screening policy.
            """,
            cls="soft-card",
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    vmax = int(max(cm_procnn.max(), cm_resnet50.max()))

    im1 = plot_confusion_matrix(axes[0], cm_procnn, "ProCNN (masked)", vmax=vmax)
    im2 = plot_confusion_matrix(axes[1], cm_resnet50, "ResNet50", vmax=vmax)

    # one colorbar per axis (clear in presentation)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    card(
        """
        <b>Visual takeaway:</b> ResNet50 shifts <b>both</b> error types downward, with the most important reduction in
        <b>false negatives (missed COVID cases)</b>.
        """,
        cls="info-card",
    )

st.markdown("---")
nav_row(top=False)
