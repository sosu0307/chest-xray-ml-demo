import streamlit as st
import pandas as pd

st.set_page_config(page_title="Deep Learning Presentation", layout="wide")

# =========================================================
# Style
# =========================================================
st.markdown(
    """
<style>
.block-container{padding-top:1.1rem;padding-bottom:1.4rem;}

.card{
  border:1px solid #cfd8e3;border-radius:12px;padding:14px 16px;background:#fff;
}
.info-card{
  border:1px solid #7ec8f8;border-radius:12px;padding:12px 14px;background:#e8fbff;
}
.success-card{
  border:1px solid #7bdcb5;border-radius:12px;padding:12px 14px;background:#eafaf3;
}
.kpi-card{
  border:1px solid #9dd8ff;border-radius:12px;padding:12px 10px;background:#edf8ff;text-align:center;
}
.kpi-label{color:#475569;font-size:.9rem;margin-bottom:4px;}
.kpi-value{color:#111827;font-weight:800;font-size:1.95rem;line-height:1.1;}
.small-muted{color:#64748b;font-size:.9rem;}

.table-wrap{width:100%;overflow-x:auto;margin-top:8px;}
div.card table.metric-table{
  width:100%!important;border-collapse:collapse!important;font-size:15px!important;line-height:1.4!important;color:#111827!important;
}
div.card table.metric-table th, div.card table.metric-table td{
  border:1px solid #d1d5db!important;padding:8px 10px!important;text-align:center!important;color:#111827!important;background:#fff!important;font-weight:600;
}
div.card table.metric-table th{
  background:#eef2f7!important;color:#0f172a!important;font-weight:800!important;
}
div.card table.metric-table tbody tr:nth-child(even) td{background:#f8fafc!important;}
div.card table.metric-table td:first-child, div.card table.metric-table th:first-child{
  text-align:left!important;min-width:150px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# In-page sections
# =========================================================
SECTIONS = [
    "Overview (What we did)",
    "ProCNN (masked) + Ablation (no mask)",
    "Transfer Learning Benchmark",
    "Final Choice (ResNet50)",
    "Live Demo",
]

if "dl_section" not in st.session_state:
    st.session_state.dl_section = SECTIONS[0]

with st.sidebar:
    st.markdown("## Navigation")
    chosen = st.radio(
        "Sections",
        SECTIONS,
        index=SECTIONS.index(st.session_state.dl_section),
        key="dl_section_radio",
    )
    st.session_state.dl_section = chosen

# =========================================================
# Header
# =========================================================
st.title("Deep Learning Models and Results")
st.caption("COVID-19 Chest X-ray Classification")

q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    if st.button("Overview ▶", use_container_width=True):
        st.session_state.dl_section = "Overview (What we did)"
        st.rerun()
with q2:
    if st.button("ProCNN ▶", use_container_width=True):
        st.session_state.dl_section = "ProCNN (masked) + Ablation (no mask)"
        st.rerun()
with q3:
    if st.button("Benchmark ▶", use_container_width=True):
        st.session_state.dl_section = "Transfer Learning Benchmark"
        st.rerun()
with q4:
    if st.button("Final Choice ▶", use_container_width=True):
        st.session_state.dl_section = "Final Choice (ResNet50)"
        st.rerun()
with q5:
    if st.button("Live Demo ▶", use_container_width=True):
        st.session_state.dl_section = "Live Demo"
        st.rerun()

st.markdown("---")

# =========================================================
# Final benchmark values (as provided)
# =========================================================
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
            "Precision": 0.7131,
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


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        out[c] = out[c].map(lambda x: "–" if pd.isna(x) else f"{x:.4f}")
    return out


section = st.session_state.dl_section

if section == "Overview (What we did)":
    st.header("Overview — our experimental setup")

    st.markdown(
        """
<div class="card">
<b>Goal:</b> COVID screening decision support on chest X-rays (binary classification).<br><br>
We evaluated three ideas:
<ol>
<li><b>Custom CNN (ProCNN)</b> trained with <b>lung masks</b> (focus on relevant regions)</li>
<li><b>Ablation</b>: same ProCNN but <b>without masks</b> (raw images)</li>
<li><b>Transfer learning benchmark</b> (ResNet18/50/101, DenseNet121, RegNet400MF; same test set)</li>
</ol>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="card">
<h4>ProCNN architecture (short)</h4>
<ul>
<li>5 conv stages (3→32→64→128→256→512)</li>
<li>Blocks: Conv + BatchNorm + ReLU + MaxPool(2×2)</li>
<li>Global feature compression via AdaptiveAvgPool(1×1)</li>
<li>Head: Linear(512→256) + ReLU + Dropout(0.5) + Linear(256→2)</li>
<li>Training focus: imbalance-aware sampling, augmentations, threshold-aware screening policy</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Quick scorecard (best checkpoints)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">ProCNN best val F1</div><div class="kpi-value">0.8852</div><div class="small-muted">(ep 22)</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">ResNet50 best val F1</div><div class="kpi-value">0.9833</div><div class="small-muted">(ep 9)</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">ResNet50 test accuracy</div><div class="kpi-value">0.9918</div><div class="small-muted">held-out</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">ResNet50 FN / FP</div><div class="kpi-value">11 / 15</div><div class="small-muted">held-out</div></div>',
            unsafe_allow_html=True,
        )

elif section == "ProCNN (masked) + Ablation (no mask)":
    st.header("ProCNN — custom baseline + ablation")

    st.subheader("A) ProCNN (masked)")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Accuracy", "0.96")
    a2.metric("COVID Precision", "0.87")
    a3.metric("COVID Recall", "0.91")
    a4.metric("COVID F1", "0.89")
    a5.metric("Macro F1", "0.93")

    st.subheader("B) Ablation: ProCNN (raw / no mask)")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Accuracy", "0.76")
    b2.metric("COVID Precision", "0.39")
    b3.metric("COVID Recall", "0.71")
    b4.metric("COVID F1", "0.51")
    b5.metric("Macro F1", "0.67")

    st.markdown(
        """
<div class="card">
<b>Interpretation:</b> Without masks, false alarms increase and overall robustness drops.
</div>
""",
        unsafe_allow_html=True,
    )

elif section == "Transfer Learning Benchmark":
    st.header("Transfer Learning Benchmark")
    st.caption("Same held-out test set (n = 3175)")

    st.markdown(
        """
<div class="info-card">
Transfer-learning backbones show excellent separability (ROC-AUC ≈ 0.999).  
Selection policy is high-stakes screening: prioritize low FN / high recall.
</div>
""",
        unsafe_allow_html=True,
    )

    show_df = fmt_table(benchmark_df)
    html_table = show_df.to_html(index=False, classes="metric-table", border=0)
    st.markdown(
        f'<div class="card"><div class="table-wrap">{html_table}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="info-card">
<b>Why ResNet50?</b> Best miss-rate behavior in our benchmark (<b>FN = 11</b>) with very strong overall performance.
</div>
""",
        unsafe_allow_html=True,
    )

elif section == "Final Choice (ResNet50)":
    st.header("Final Choice — ResNet50")
    st.markdown(
        """
<div class="card">
<b>Decision policy:</b> screening is high-stakes, so we prioritize <b>Recall / FN</b> over marginal gains in accuracy.<br><br>
<b>Final decision threshold:</b> <b>0.40</b> (recall/FN-prioritized).<br>
<b>Test set size:</b> n = 3175.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ResNet50 test FN", "11")
    c2.metric("ResNet50 test FP", "15")
    c3.metric("ResNet50 test Recall", "0.9797")
    c4.metric("ResNet50 test F1", "0.9761")

    st.markdown(
        '<div class="success-card">🏆 <b>Winner:</b> ResNet50 chosen for strongest high-stakes behavior (low FN, high recall).</div>',
        unsafe_allow_html=True,
    )

elif section == "Live Demo":
    st.header("Live Demo")
    st.markdown(
        '<div class="card">Interactive threshold demo on borderline samples + Grad-CAM inspection.</div>',
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Open Inference Demo ▶", use_container_width=True):
            st.switch_page("pages/3_Inference_Demo.py")
    with d2:
        if st.button("Open Grad-CAM Demo ▶", use_container_width=True):
            st.switch_page("pages/4_Gradcam_Demo.py")

st.caption("For educational and demonstration use only. Not for clinical diagnosis.")
