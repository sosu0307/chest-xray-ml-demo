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
    PROJECT_ROOT / "data_bin" / "test" / "noncovid" / "images" / "Normal-1809.png"
)

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "results" / "resnet50_best.pt",
    PROJECT_ROOT / "models" / "resnet50_best.pt",
    PROJECT_ROOT / "models" / "results" / "resnet50_last.pt",
    PROJECT_ROOT / "models" / "resnet50_last.pt",
]


def find_model_path():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


MODEL_PATH = find_model_path()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Grad-CAM Demo", layout="wide")
st.title("Grad-CAM Demo — ResNet50")
st.caption("Fixed demo images (no auto-reload / no auto-selection).")

n1, n2 = st.columns(2)
with n1:
    if st.button("← Back to Inference Demo", use_container_width=True):
        st.switch_page("pages/3_Inference_Demo.py")
with n2:
    if st.button("Back to Presentation", use_container_width=True):
        st.switch_page("0_Deeplearning_Presentation.py")

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
st.caption(f"Using model: {MODEL_PATH}")

# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------
advanced = st.toggle("Advanced Grad-CAM controls", value=False)

if advanced:
    target_option = st.selectbox(
        "Target class",
        ["Predicted class", "COVID", "Non-COVID"],
        index=0,
    )
else:
    target_option = "Predicted class"
    st.caption("Target class: Predicted class (recommended)")

# Fixed sample chooser
st.subheader("Fixed demo samples")
s1, s2 = st.columns(2)

if "sel_class" not in st.session_state:
    st.session_state.sel_class = "covid"

with s1:
    c_img = Image.open(COVID_IMG_PATH).convert("RGB")
    st.image(c_img, width=230, caption=f"{COVID_IMG_PATH.name} (covid)")
    if st.button("Use COVID sample", key="use_covid"):
        st.session_state.sel_class = "covid"

with s2:
    n_img = Image.open(NONCOVID_IMG_PATH).convert("RGB")
    st.image(n_img, width=230, caption=f"{NONCOVID_IMG_PATH.name} (noncovid)")
    if st.button("Use Non-COVID sample", key="use_noncovid"):
        st.session_state.sel_class = "noncovid"

sel_class = st.session_state.sel_class
if sel_class == "covid":
    sel_path = COVID_IMG_PATH
    true_cls = "covid"
else:
    sel_path = NONCOVID_IMG_PATH
    true_cls = "noncovid"

pil = Image.open(sel_path).convert("RGB")

if target_option == "Predicted class":
    t_idx = None
elif target_option == "COVID":
    t_idx = 1
else:
    t_idx = 0

orig, heatmap, probs = gradcam_for_resnet50(model, pil, class_idx=t_idx)
overlay = make_overlay(orig, heatmap)
pred = "covid" if int(np.argmax(probs)) == 1 else "noncovid"

st.divider()
st.markdown(
    f"**Selected:** `{sel_path.name}` | **True:** `{true_cls}` | **Predicted:** `{pred}`"
)

v1, v2, v3 = st.columns(3)
with v1:
    st.markdown("### Original")
    st.image(orig, width=320)

with v2:
    st.markdown("### Heatmap")
    heat_rgb = (plt.get_cmap("jet")(heatmap)[..., :3] * 255).astype(np.uint8)
    st.image(heat_rgb, width=320)

with v3:
    st.markdown("### Overlay")
    st.image(overlay, width=320)

m1, m2 = st.columns(2)
with m1:
    st.metric("P(Non-COVID)", f"{probs[0]:.3f}")
with m2:
    st.metric("P(COVID)", f"{probs[1]:.3f}")

st.caption("For educational and demonstration use only. Not for clinical diagnosis.")
