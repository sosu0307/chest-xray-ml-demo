# src/streamlit/pages/4_Gradcam_Demo.py
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Paths / config
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
STREAMLIT_DIR = CURRENT_FILE.parents[1]  # src/streamlit
PROJECT_ROOT = STREAMLIT_DIR.parent.parent  # repo root

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COVID_IMG_PATH = (
    PROJECT_ROOT / "data_bin" / "test" / "covid" / "images" / "COVID-1043.png"
)
NONCOVID_IMG_PATH = (
    PROJECT_ROOT / "data_bin" / "test" / "noncovid" / "images" / "Normal-2202.png"
)

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "results" / "resnet50_best.pt",
    PROJECT_ROOT / "models" / "resnet50_best.pt",
    PROJECT_ROOT / "models" / "results" / "resnet50_last.pt",
    PROJECT_ROOT / "models" / "resnet50_last.pt",
]

# Page links (adjust if your filenames differ)
PAGE_INFERENCE_DEMO = "pages/3_3_inference_demo.py"
PAGE_CONCLUSION = "pages/4_conclusion_questions.py"


def find_model_path():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


MODEL_PATH = find_model_path()

# =========================================================
# Style (aligned with inference page)
# =========================================================
st.markdown(
    """
<style>
.block-container{
    padding-top: 1.0rem !important;
    padding-bottom: 1.2rem !important;
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
    min-height:92px;
}
.kpi-label{
    color:#5f6b7a;
    font-size:0.80rem;
    margin-bottom:4px;
    font-weight:600;
}
.kpi-value{
    font-size:1.18rem;
    font-weight:700;
    line-height:1.15;
    color:#0f172a;
}
.kpi-sub{
    color:#5f6b7a;
    font-size:0.82rem;
    margin-top:4px;
}

/* Sample selector cards */
.sample-choice{
    border:1px solid #d9e1ea;
    border-radius:14px;
    background:#ffffff;
    padding:12px;
}
.sample-choice.selected{
    border:1px solid #93c5fd;
    background:#f7fbff;
    box-shadow: inset 0 0 0 1px #dbeafe;
}
.sample-caption{
    text-align:center;
    color:#5f6b7a;
    font-size:0.85rem;
    margin-top:6px;
}
.badge{
    display:inline-block;
    border-radius:999px;
    padding:2px 8px;
    font-size:0.78rem;
    font-weight:700;
    margin-bottom:8px;
}
.badge-blue{
    background:#eff6ff;
    color:#1d4ed8;
    border:1px solid #bfdbfe;
}
.badge-green{
    background:#ecfdf3;
    color:#166534;
    border:1px solid #bbf7d0;
}
.badge-gray{
    background:#f3f4f6;
    color:#374151;
    border:1px solid #e5e7eb;
}

/* Result image panels */
.result-panel{
    border:1px solid #d9e1ea;
    border-radius:14px;
    background:#ffffff;
    padding:10px 10px 6px 10px;
}
.panel-title{
    font-weight:700;
    font-size:1.00rem;
    margin:0 0 8px 0;
    color:#111827;
}

/* Buttons */
div.stButton > button{
    min-height: 52px !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
div.stButton > button p{
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


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------
def safe_switch(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        st.warning(
            f"Could not open page: `{page_path}`. Check file name/path in `src/streamlit/pages/`."
        )


def card(html: str, cls: str = "card"):
    st.markdown(f"<div class='{cls}'>{html}</div>", unsafe_allow_html=True)


def kpi(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{title}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("Grad-CAM Demo — ResNet50")
st.caption(
    "Interpretability demo for a fixed COVID and non-COVID sample. "
    "Educational demonstration only — not for clinical diagnosis."
)

# Top navigation
n1, n2 = st.columns(2)
with n1:
    if st.button("⬅ Inference Demo", use_container_width=True, key="nav_top_infer"):
        safe_switch(PAGE_INFERENCE_DEMO)
with n2:
    if st.button(
        "Conclusion & Questions ➜", use_container_width=True, key="nav_top_conc"
    ):
        safe_switch(PAGE_CONCLUSION)

st.divider()


# ---------------------------------------------------------
# Model + preprocessing
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    if model_path is None:
        raise FileNotFoundError("Model not found in expected paths.")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Checkpoint format not recognized.")

    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


TFM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def preprocess_pil(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return TFM(img).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------
def gradcam_for_resnet50(
    model: nn.Module, pil_img: Image.Image, class_idx: int | None = None
):
    x = preprocess_pil(pil_img)
    orig_np = np.array(pil_img.convert("RGB"))

    acts, grads = [], []
    target_layer = model.layer4[-1]

    def fwd_hook(_m, _i, o):
        acts.append(o)

    def bwd_hook(_m, _gi, go):
        grads.append(go[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())

    score = logits[:, class_idx].sum()
    score.backward()

    a = acts[0]
    g = grads[0]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = (w * a).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam,
        size=(orig_np.shape[0], orig_np.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    cam = cam[0, 0].detach().cpu().numpy()

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    h1.remove()
    h2.remove()
    return orig_np, cam.astype(np.float32), probs


def make_overlay(orig_np: np.ndarray, heatmap: np.ndarray):
    alpha = 0.45
    heat_rgb = plt.get_cmap("jet")(heatmap)[..., :3]
    base = orig_np.astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * heat_rgb
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------
# Guards + load fixed resources
# ---------------------------------------------------------
if MODEL_PATH is None:
    st.error(
        "Model not found. Checked:\n" + "\n".join(str(p) for p in MODEL_CANDIDATES)
    )
    st.stop()

if not COVID_IMG_PATH.exists():
    st.error(f"COVID image not found:\n{COVID_IMG_PATH}")
    st.stop()

if not NONCOVID_IMG_PATH.exists():
    st.error(f"Non-COVID image not found:\n{NONCOVID_IMG_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------
# Intro / page purpose
# ---------------------------------------------------------
card(
    """
    <b>What this page shows:</b> Grad-CAM highlights image regions that influenced the model’s decision for the selected sample.<br>
    <b>Important:</b> This is an <b>interpretability aid</b> for demonstration and discussion — not proof of clinical validity.
    """,
    cls="info-card",
)

# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------
st.subheader("Grad-CAM Controls")

c_left, c_right = st.columns([1.2, 1.8], gap="large")
with c_left:
    advanced = st.toggle("Advanced Grad-CAM controls", value=False)

with c_right:
    if advanced:
        target_option = st.selectbox(
            "Target class",
            ["Predicted class", "COVID", "Non-COVID"],
            index=0,
        )
    else:
        target_option = "Predicted class"
        card(
            "Target class is fixed to <b>Predicted class</b> (recommended for live demo).",
            cls="soft-card",
        )

# ---------------------------------------------------------
# Sample selection
# ---------------------------------------------------------
st.subheader("Sample Selection")

if "sel_class" not in st.session_state:
    st.session_state.sel_class = "covid"

s1, s2 = st.columns(2, gap="large")

with s1:
    with Image.open(COVID_IMG_PATH) as _im:
        c_img = _im.convert("RGB")

    css_cls = (
        "sample-choice selected"
        if st.session_state.sel_class == "covid"
        else "sample-choice"
    )
    st.markdown(f"<div class='{css_cls}'>", unsafe_allow_html=True)
    st.markdown(
        "<span class='badge badge-blue'>COVID sample</span>", unsafe_allow_html=True
    )
    st.image(c_img, width=260)
    st.markdown(
        f"<div class='sample-caption'>{COVID_IMG_PATH.name} (covid)</div>",
        unsafe_allow_html=True,
    )
    if st.button("Use COVID sample", use_container_width=True, key="use_covid"):
        st.session_state.sel_class = "covid"
    st.markdown("</div>", unsafe_allow_html=True)

with s2:
    with Image.open(NONCOVID_IMG_PATH) as _im:
        n_img = _im.convert("RGB")

    css_cls = (
        "sample-choice selected"
        if st.session_state.sel_class == "noncovid"
        else "sample-choice"
    )
    st.markdown(f"<div class='{css_cls}'>", unsafe_allow_html=True)
    st.markdown(
        "<span class='badge badge-gray'>Non-COVID sample</span>", unsafe_allow_html=True
    )
    st.image(n_img, width=260)
    st.markdown(
        f"<div class='sample-caption'>{NONCOVID_IMG_PATH.name} (noncovid)</div>",
        unsafe_allow_html=True,
    )
    if st.button("Use Non-COVID sample", use_container_width=True, key="use_noncovid"):
        st.session_state.sel_class = "noncovid"
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resolve selection and target
# ---------------------------------------------------------
sel_class = st.session_state.sel_class
if sel_class == "covid":
    sel_path = COVID_IMG_PATH
    true_cls = "covid"
else:
    sel_path = NONCOVID_IMG_PATH
    true_cls = "noncovid"

with Image.open(sel_path) as _im:
    pil = _im.convert("RGB")

if target_option == "Predicted class":
    t_idx = None
elif target_option == "COVID":
    t_idx = 1
else:
    t_idx = 0

orig, heatmap, probs = gradcam_for_resnet50(model, pil, class_idx=t_idx)
overlay = make_overlay(orig, heatmap)
pred = "covid" if int(np.argmax(probs)) == 1 else "noncovid"

target_text = target_option if advanced else "Predicted class"

st.divider()

# ---------------------------------------------------------
# Selection summary
# ---------------------------------------------------------
card(
    f"""
    <b>Selected sample:</b> {sel_path.name}<br>
    <b>True label:</b> {true_cls} &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>Predicted label:</b> {pred} &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>Grad-CAM target:</b> {target_text}
    """,
    cls="soft-card",
)

# ---------------------------------------------------------
# Visualization panels
# ---------------------------------------------------------
st.subheader("Grad-CAM Visualization")

v1, v2, v3 = st.columns(3, gap="large")

with v1:
    st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Original</div>", unsafe_allow_html=True)
    st.image(orig, use_column_width=380)
    st.markdown("</div>", unsafe_allow_html=True)

with v2:
    st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Heatmap</div>", unsafe_allow_html=True)
    heat_rgb = (plt.get_cmap("jet")(heatmap)[..., :3] * 255).astype(np.uint8)
    st.image(heat_rgb, use_column_width=380)
    st.markdown("</div>", unsafe_allow_html=True)

with v3:
    st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Overlay</div>", unsafe_allow_html=True)
    st.image(overlay, use_column_width=380)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Probabilities (KPI style)
# ---------------------------------------------------------
p1, p2 = st.columns(2, gap="large")
with p1:
    kpi("P(Non-COVID)", f"{probs[0]:.3f}", "model output probability")
with p2:
    kpi("P(COVID)", f"{probs[1]:.3f}", "model output probability")

# ---------------------------------------------------------
# Takeaway / caution (instead of plain caption only)
# ---------------------------------------------------------
st.markdown(
    """
<div class="success-card">
  <h4 style="margin:0 0 8px 0;">Key Takeaways (Interpretability / Reliability Check)</h4>
  <ul style="margin:0; padding-left: 1.05rem;">
    <li><b>Grad-CAM is used as a supportive interpretability check</b> to inspect where the model focuses in the image.</li>
    <li>For a plausible result, attention should be concentrated mainly in <b>relevant chest/lung regions</b> rather than unrelated background areas.</li>
    <li>The visualization is helpful for <b>discussion, sanity-checking, and debugging</b> model behavior.</li>
    <li>Grad-CAM <b>does not prove clinical validity</b> and does not establish causal reasoning.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='small-muted'>Educational demo only. Grad-CAM visualizations support interpretation but are not a clinical validation method.</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------
# Bottom navigation (same layout style as other pages)
# ---------------------------------------------------------
b1, b2 = st.columns(2)
with b1:
    if st.button("⬅ Inference Demo", use_container_width=True, key="nav_bottom_infer"):
        safe_switch(PAGE_INFERENCE_DEMO)
with b2:
    if st.button(
        "Conclusion & Questions ➜", use_container_width=True, key="nav_bottom_conc"
    ):
        safe_switch(PAGE_CONCLUSION)
