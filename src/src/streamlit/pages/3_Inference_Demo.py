import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# -----------------------------------
# path import robustness
# -----------------------------------
CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parents[1]  # src/streamlit
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

from app_paths import MODEL_PATH  # noqa: E402

DEMO_CSV = STREAMLIT_DIR / "data" / "demo4_borderline.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Inference Demo", layout="wide")

# =========================================================
# Unified style (same family as your other pages)
# =========================================================
st.markdown(
    """
<style>
/* ---------- Layout ---------- */
.block-container{
    padding-top: 1.8rem !important;
    padding-bottom: 1.4rem !important;
    max-width: 1200px;
}

/* ---------- Headings ---------- */
h1, .stMarkdown h1{
    margin-top: 0 !important;
    line-height: 1.2 !important;
}
h2, .stMarkdown h2, h3, .stMarkdown h3{
    line-height: 1.25 !important;
}

/* ---------- Cards ---------- */
.card{
    border:1px solid #cfd8e3;
    border-radius:12px;
    padding:12px 14px;
    background:#ffffff;
}
.info-card{
    border:1px solid #7ec8f8;
    border-radius:12px;
    padding:12px 14px;
    background:#e8fbff;
}

/* ---------- KPI mini cards ---------- */
.kpi-card{
    border:1px solid #9ed8fb;
    border-radius:12px;
    padding:10px 12px;
    background:#edf8ff;
    text-align:left;
    min-height:94px;
}
.kpi-label{
    font-size:.86rem;
    color:#37526b;
    margin-bottom:4px;
    font-weight:600;
}
.kpi-value{
    font-size:1.55rem;
    font-weight:800;
    color:#0f172a;
    line-height:1.1;
}
.small-muted{
    color:#5b7288;
    font-size:.88rem;
}

/* ---------- Pred result coloring ---------- */
.pred-ok{ color:#16a34a; font-weight:700; }
.pred-bad{ color:#dc2626; font-weight:700; }
.mono{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

/* ---------- spacing helpers ---------- */
.btn-row-gap{
    margin-top:0.15rem;
    margin-bottom:0.55rem;
}
.section-gap{
    margin-top:0.35rem;
    margin-bottom:0.35rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Header
# =========================================================
st.title("Inference Demo — ResNet50 (4 Borderline Samples)")
st.caption("Educational demo only. Not for clinical diagnosis.")

# =========================================================
# Top Navigation (UNIFIED)
# =========================================================
n1, n2, n3 = st.columns(3)
with n1:
    if st.button("← Back to Training Curves", use_container_width=True):
        st.switch_page("pages/2_Training_Curves.py")
with n2:
    if st.button("Back to Presentation", use_container_width=True):
        st.switch_page("0_Deeplearning_Presentation.py")
with n3:
    if st.button("Next ▶ Grad-CAM Demo", use_container_width=True):
        st.switch_page("pages/4_Gradcam_Demo.py")


# =========================================================
# Model + helpers
# =========================================================
@st.cache_resource
def load_model_and_threshold():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        best_threshold = float(ckpt.get("best_threshold", 0.40))
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        best_threshold = 0.40
    else:
        raise RuntimeError("Checkpoint format not recognized.")

    model.to(DEVICE).eval()
    return model, best_threshold


@st.cache_data
def load_demo4(csv_path: Path):
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    needed = {"true_class", "filename", "abs_path", "prob_covid", "distance"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df


def preprocess_image(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def predict_covid_prob(model, img: Image.Image) -> float:
    x = preprocess_image(img).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[1])


# =========================================================
# Guards
# =========================================================
if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

model, default_threshold = load_model_and_threshold()

df_all = load_demo4(DEMO_CSV)
if df_all.empty:
    st.warning(
        "demo4_borderline.csv missing/invalid.\n\n"
        "Please run:\n"
        "python src/streamlit/prepare_demo4_csv.py"
    )
    st.stop()

# =========================================================
# Controls row
# =========================================================
c1, c2 = st.columns([2, 1])
with c1:
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, float(default_threshold), 0.01
    )
with c2:
    class_filter = st.selectbox("Class filter", ["all", "covid", "noncovid"], index=0)

# Optional concise policy info (same tone as other pages)
st.markdown(
    f"""
<div class="info-card">
  <b>Decision policy:</b> high-stakes screening focuses on reducing missed positives (FN).<br>
  <b>Operating threshold shown live:</b> <span class="mono">{threshold:.2f}</span>
</div>
""",
    unsafe_allow_html=True,
)

# Filtered data
df = df_all.copy()
if class_filter != "all":
    df = df[df["true_class"] == class_filter].copy()
df = df.reset_index(drop=True)

# =========================================================
# Quick summary KPIs for current view
# =========================================================
# Recompute live predictions for summary
pred_rows = []
for _, row in df.iterrows():
    p = Path(row["abs_path"])
    if not p.exists():
        continue
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        continue
    prob = predict_covid_prob(model, img)
    pred = "covid" if prob >= threshold else "noncovid"
    pred_rows.append((row["true_class"], pred))

n_total = len(pred_rows)
n_correct = sum(1 for t, p in pred_rows if t == p)
n_wrong = n_total - n_correct

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Samples shown</div>
  <div class="kpi-value">{n_total}</div>
  <div class="small-muted">filtered view</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Threshold</div>
  <div class="kpi-value">{threshold:.2f}</div>
  <div class="small-muted">live decision cut-off</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Correct</div>
  <div class="kpi-value">{n_correct}</div>
  <div class="small-muted">on shown samples</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">Changed-risk errors</div>
  <div class="kpi-value">{n_wrong}</div>
  <div class="small-muted">on shown samples</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

# =========================================================
# Sample grid (4 columns)
# =========================================================
cols = st.columns(4)

for i, row in df.iterrows():
    with cols[i % 4]:
        p = Path(row["abs_path"])
        if not p.exists():
            st.error(f"Missing file: {p}")
            continue

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            st.error(f"Could not open {p.name}: {e}")
            continue

        prob = predict_covid_prob(model, img)
        pred = "covid" if prob >= threshold else "noncovid"
        true_cls = row["true_class"]
        ok = pred == true_cls

        st.image(img, width=220, caption=p.name)

        status_html = (
            f"<span class='pred-ok'>✅ {pred}</span>"
            if ok
            else f"<span class='pred-bad'>❌ {pred}</span>"
        )

        st.markdown(
            f"""
<div class="card" style="margin-top:6px;">
  <div><b>True:</b> <span class="mono">{true_cls}</span></div>
  <div><b>p(covid):</b> <span class="mono">{prob:.3f}</span></div>
  <div><b>Pred:</b> {status_html}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Only 4 fixed borderline samples (2 covid + 2 noncovid) from test/images. "
    "Use threshold slider to show label flips near the decision boundary."
)
