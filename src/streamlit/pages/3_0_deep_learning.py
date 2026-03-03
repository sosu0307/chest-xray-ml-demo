import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Deep Learning")

# =========================================================
# Paths (robust if script is inside src/streamlit/pages/)
# =========================================================
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "pages":
    STREAMLIT_DIR = _THIS_FILE.parent.parent  # src/streamlit
else:
    STREAMLIT_DIR = _THIS_FILE.parent  # src/streamlit

# =========================================================
# Fixed page paths (chapter 3 structure only)
# =========================================================
PAGE_MODEL_COMPARISON = "pages/3_1_model_comparison.py"
PAGE_TRAINING_CURVES = "pages/3_2_training_curves.py"
PAGE_INFERENCE_DEMO = "pages/3_3_inference_demo.py"
PAGE_GRADCAM_DEMO = "pages/3_4_gradcam_demo.py"

# =========================================================
# Style
# =========================================================
st.markdown(
    """
<style>
.block-container{
    padding-top: 1.0rem;
    padding-bottom: 1.1rem;
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

/* Delta KPI cards */
.delta-card{
    border-radius:14px;
    padding:12px 10px;
    text-align:center;
    height:100%;
}
.delta-neutral{
    border:1px solid #dfe7ef;
    background:#fbfdff;
}
.delta-good{
    border:1px solid #b7e3cc;
    background:#eefaf4;
}
.delta-title{
    color:#5f6b7a;
    font-size:0.80rem;
    margin-bottom:4px;
}
.delta-value{
    font-size:1.06rem;
    font-weight:700;
    line-height:1.15;
}
.delta-sub{
    color:#5f6b7a;
    font-size:0.82rem;
    margin-top:3px;
}

/* Small flow cards for start tab */
.flow-step{
    border:1px solid #dfe7ef;
    background:#fbfdff;
    border-radius:12px;
    padding:10px 12px;
    min-height:92px;
    margin-bottom:10px;
}
.flow-step-title{
    font-weight:700;
    font-size:0.93rem;
    margin-bottom:3px;
}
.flow-step-sub{
    color:#5f6b7a;
    font-size:0.84rem;
    line-height:1.25;
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
def card(html: str, cls: str = "card"):
    st.markdown(f"<div class='{cls}'>{html}</div>", unsafe_allow_html=True)


def kpi(title: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def delta_kpi(title: str, value: str, sub: str = "", good: bool = False):
    cls = "delta-card delta-good" if good else "delta-card delta-neutral"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="delta-title">{title}</div>
            <div class="delta-value">{value}</div>
            <div class="delta-sub">{sub}</div>
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


def render_subpage_nav(prefix: str):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Model Comparison", use_container_width=True, key=f"{prefix}_mc"):
            safe_switch(PAGE_MODEL_COMPARISON)
    with c2:
        if st.button("Training Curves", use_container_width=True, key=f"{prefix}_tc"):
            safe_switch(PAGE_TRAINING_CURVES)
    with c3:
        if st.button("Inference Demo", use_container_width=True, key=f"{prefix}_id"):
            safe_switch(PAGE_INFERENCE_DEMO)
    with c4:
        if st.button("Grad-CAM Demo", use_container_width=True, key=f"{prefix}_gc"):
            safe_switch(PAGE_GRADCAM_DEMO)


# =========================================================
# Data (Report 2 summary)
# =========================================================
procnn_compare_df = pd.DataFrame(
    [
        {
            "Variant": "ProCNN (with lung masks)",
            "Accuracy": 0.96,
            "COVID Precision": 0.87,
            "COVID Recall": 0.91,
            "COVID F1": 0.89,
            "Macro F1": 0.93,
        },
        {
            "Variant": "ProCNN (raw X-rays / no mask)",
            "Accuracy": 0.76,
            "COVID Precision": 0.39,
            "COVID Recall": 0.71,
            "COVID F1": 0.51,
            "Macro F1": 0.67,
        },
    ]
)

delta_row = {
    "Variant": "Delta (masked - no mask)",
    "Accuracy": round(0.96 - 0.76, 2),
    "COVID Precision": round(0.87 - 0.39, 2),
    "COVID Recall": round(0.91 - 0.71, 2),
    "COVID F1": round(0.89 - 0.51, 2),
    "Macro F1": round(0.93 - 0.67, 2),
}
procnn_compare_display = pd.concat(
    [procnn_compare_df, pd.DataFrame([delta_row])], ignore_index=True
)

benchmark_df = pd.DataFrame(
    [
        {
            "Model": "ResNet18",
            "Accuracy": 0.9931,
            "Precision": 0.9815,
            "Recall": 0.9779,
            "F1": 0.9797,
            "ROC-AUC": 0.9993,
            "TN": 2623,
            "FP": 10,
            "FN": 12,
            "TP": 530,
        },
        {
            "Model": "ResNet50",
            "Accuracy": 0.9918,
            "Precision": 0.9725,
            "Recall": 0.9797,
            "F1": 0.9761,
            "ROC-AUC": 0.9994,
            "TN": 2618,
            "FP": 15,
            "FN": 11,
            "TP": 531,
        },
        {
            "Model": "ResNet101",
            "Accuracy": 0.9909,
            "Precision": 0.9689,
            "Recall": 0.9779,
            "F1": 0.9734,
            "ROC-AUC": 0.9992,
            "TN": 2616,
            "FP": 17,
            "FN": 12,
            "TP": 530,
        },
        {
            "Model": "DenseNet121",
            "Accuracy": 0.9915,
            "Precision": 0.9813,
            "Recall": 0.9686,
            "F1": 0.9749,
            "ROC-AUC": 0.9989,
            "TN": 2623,
            "FP": 10,
            "FN": 17,
            "TP": 525,
        },
        {
            "Model": "RegNet400MF",
            "Accuracy": 0.9222,
            "Precision": 0.71315,
            "Recall": 0.9096,
            "F1": 0.7997,
            "ROC-AUC": None,
            "TN": 2435,
            "FP": 198,
            "FN": 49,
            "TP": 493,
        },
    ]
)

best_acc_model = benchmark_df.loc[benchmark_df["Accuracy"].idxmax(), "Model"]
best_f1_model = benchmark_df.loc[benchmark_df["F1"].idxmax(), "Model"]
best_recall_model = benchmark_df.loc[benchmark_df["Recall"].idxmax(), "Model"]
lowest_fn_model = benchmark_df.loc[benchmark_df["FN"].idxmin(), "Model"]

benchmark_table = benchmark_df.copy()
metric_cols_benchmark = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
benchmark_table[metric_cols_benchmark] = benchmark_table[metric_cols_benchmark].apply(
    pd.to_numeric, errors="coerce"
)
benchmark_table = benchmark_table[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
]

# =========================================================
# Header + top nav
# =========================================================
st.title("🩻 Deep Learning")
st.caption(
    "Chapter overview in presentation mode: chapter flow, evaluation policy, approaches, ablation, and benchmark evidence."
)

render_subpage_nav(prefix="top")
st.markdown("---")

# =========================================================
# Presentation-mode Tabs (reduces scrolling)
# =========================================================
tab_start, tab_approaches, tab_ablation, tab_benchmark = st.tabs(
    [
        "Chapter Start",
        "Approaches & ProCNN Design",
        "Ablation (Mask vs No Mask)",
        "Transfer Learning Benchmark",
    ]
)

# ---------------------------------------------------------
# Tab 1: Chapter Start (slim + speaking flow only)
# ---------------------------------------------------------
with tab_start:

    st.subheader("Evaluation policy")
    card(
        """
        <h4 style="margin:0 0 8px 0;">High-Stakes Policy (Recall-first)</h4>
        <ul style="margin:0; padding-left: 1.05rem;">
          <li><b>Primary risk to avoid:</b> false negatives (missed COVID cases)</li>
          <li>We prioritize <b>COVID recall</b> and <b>FN behavior</b> (not only a single score like accuracy)</li>
          <li>This policy also motivates <b>threshold tuning</b> and the later demo pages</li>
        </ul>
        """,
        cls="success-card",
    )

    st.subheader("Chapter roadmap")
    card(
        """
    <ul style="margin:0; padding-left: 1.05rem;">
      <li><b>Screening objective:</b> recall-first policy (reduce missed COVID cases)</li>
      <li><b>Approaches:</b> ProCNN baseline and transfer-learning benchmark</li>
      <li><b>Ablation:</b> mask vs. no-mask preprocessing impact</li>
      <li><b>Model selection:</b> metrics + confusion-matrix evidence</li>
      <li><b>Evidence pages:</b> training curves, inference demo, Grad-CAM demo</li>
    </ul>
    """,
        cls="soft-card",
    )
# ---------------------------------------------------------
# Tab 2: Approaches + ProCNN Design
# ---------------------------------------------------------
with tab_approaches:
    st.subheader("Modeling Approaches")

    a1, a2, a3 = st.columns(3, gap="large")
    with a1:
        card(
            """
            <div style="min-height: 220px;">
              <h5 style="margin:0 0 8px 0;">ProCNN (with lung masks)</h5>
              <ul style="margin:0; padding-left:1.05rem;">
                <li>Custom CNN baseline</li>
                <li>Trained on masked lung regions</li>
                <li>Reduces background influence</li>
                <li>Used as interpretable custom reference</li>
              </ul>
            </div>
            """,
            cls="card",
        )

    with a2:
        card(
            """
            <div style="min-height: 220px;">
              <h5 style="margin:0 0 8px 0;">ProCNN Ablation (without masks)</h5>
              <ul style="margin:0; padding-left:1.05rem;">
                <li>Same ProCNN architecture</li>
                <li>Trained on raw X-rays (no masks)</li>
                <li>Quantifies masking benefit</li>
                <li>Supports preprocessing design choice</li>
              </ul>
            </div>
            """,
            cls="card",
        )

    with a3:
        card(
            """
            <div style="min-height: 220px;">
              <h5 style="margin:0 0 8px 0;">Transfer Learning Benchmark</h5>
              <ul style="margin:0; padding-left:1.05rem;">
                <li>Same held-out test split across models</li>
                <li>ResNet18 / ResNet50 / ResNet101</li>
                <li>DenseNet121 / RegNet400MF</li>
                <li>Final candidate selection basis</li>
              </ul>
            </div>
            """,
            cls="card",
        )

    st.markdown("### ProCNN Architecture and Training Policy")
    card(
        """
        <h5 style="margin:0 0 8px 0;">ProCNN Design Overview</h5>
        <ul style="margin:0; padding-left:1.05rem;">
          <li><b>Architecture:</b> 5 convolution stages with increasing channels (<b>32 → 64 → 128 → 256 → 512</b>)</li>
          <li><b>Core blocks:</b> Conv + BatchNorm + ReLU with spatial downsampling</li>
          <li><b>Feature aggregation:</b> Adaptive Global Average Pooling before classification</li>
          <li><b>Classifier head:</b> compact fully connected head (<b>512 → 256 → 2</b>) with dropout regularization</li>
          <li><b>Training setup:</b> augmentation + imbalance-aware sampling and F1-oriented control</li>
          <li><b>Selection policy:</b> screening-oriented threshold (around <b>0.40</b>) to support low-FN decisions</li>
        </ul>
        """,
        cls="info-card",
    )

# ---------------------------------------------------------
# Tab 3: Ablation
# ---------------------------------------------------------
with tab_ablation:
    st.subheader("Ablation: ProCNN With Mask vs Without Mask")

    st.dataframe(
        procnn_compare_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.2f"),
            "COVID Precision": st.column_config.NumberColumn(format="%.2f"),
            "COVID Recall": st.column_config.NumberColumn(format="%.2f"),
            "COVID F1": st.column_config.NumberColumn(format="%.2f"),
            "Macro F1": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # Quick delta KPIs for presentation
    d1, d2, d3 = st.columns(3)
    with d1:
        delta_kpi("Δ Accuracy", "+0.20", "masked - no mask", good=True)
    with d2:
        delta_kpi("Δ COVID F1", "+0.38", "masked - no mask", good=True)
    with d3:
        delta_kpi("Δ COVID Precision", "+0.48", "masked - no mask", good=True)

    card(
        """
        <h4 style="margin:0 0 8px 0;">Ablation Takeaway</h4>
        <ul style="margin:0; padding-left:1.05rem;">
          <li>Masked ProCNN outperforms the unmasked variant across all reported metrics</li>
          <li>Largest gains appear in <b>COVID Precision</b> and <b>COVID F1</b></li>
          <li>This supports keeping the masked pipeline as the stronger custom baseline</li>
        </ul>
        """,
        cls="success-card",
    )

# ---------------------------------------------------------
# Tab 4: Benchmark
# ---------------------------------------------------------
# ---------------------------------------------------------
# Tab 4: Benchmark
# ---------------------------------------------------------
with tab_benchmark:
    st.subheader("Transfer Learning Benchmark")

    card(
        """
        <h4 style="margin:0 0 8px 0;">Fair-comparison note</h4>
        <ul style="margin:0; padding-left:1.05rem;">
          <li>All models are evaluated on the <b>same held-out test set (n = 3175)</b></li>
          <li>Metric columns (<b>Accuracy / Precision / Recall / F1 / ROC-AUC</b>) are directly comparable</li>
          <li>Confusion-count columns (<b>TN / FP / FN / TP</b>) are also directly comparable</li>
        </ul>
        """,
        cls="info-card",
    )

    # ResNet50-focused selection note (instead of bottom KPI cards)
    card(
        """
        <h4 style="margin:0 0 8px 0;">Recall-first selection focus for the demo pipeline</h4>
        <ul style="margin:0; padding-left:1.05rem;">
          <li><b>ResNet50</b> is highlighted because the project uses a <b>recall-first screening policy</b></li>
          <li>It shows the <b>lowest FN (missed COVID cases)</b> in this benchmark table</li>
          <li>It keeps <b>strong overall performance</b> while supporting the later demo and reliability pages</li>
        </ul>
        <div class="small-muted" style="margin-top:8px;">
          Note: We do not select the final model by a single metric winner rule (e.g., only accuracy or only F1).
        </div>
        """,
        cls="success-card",
    )

    # --- Highlight ResNet50 row in the benchmark table ---
    benchmark_table_view = benchmark_table.copy()

    def _highlight_resnet50_row(row):
        if row["Model"] == "ResNet50":
            # subtle green highlight (fits recall-first / final selection emphasis)
            style = (
                "background-color: #eefaf4; "
                "font-weight: 600; "
                "border-top: 1px solid #b7e3cc; "
                "border-bottom: 1px solid #b7e3cc;"
            )
            return [style] * len(row)
        return [""] * len(row)

    styled_benchmark = benchmark_table_view.style.apply(
        _highlight_resnet50_row, axis=1
    ).format(
        {
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "ROC-AUC": lambda x: "—" if pd.isna(x) else f"{x:.4f}",
            "TN": "{:.0f}",
            "FP": "{:.0f}",
            "FN": "{:.0f}",
            "TP": "{:.0f}",
        },
        na_rep="—",
    )

    st.dataframe(
        styled_benchmark,
        use_container_width=True,
        hide_index=True,
    )
