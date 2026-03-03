from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# =========================================================
# Paths (robust for src/streamlit/pages/)
# =========================================================
CURRENT_FILE = Path(__file__).resolve()

if CURRENT_FILE.parent.name == "pages":
    STREAMLIT_DIR = CURRENT_FILE.parent.parent  # src/streamlit
else:
    STREAMLIT_DIR = CURRENT_FILE.parent  # src/streamlit

PROJECT_ROOT = STREAMLIT_DIR.parent.parent  # repo root

# Chapter 3 page paths
PAGE_DEEP_LEARNING = "pages/3_0_deep_learning.py"
PAGE_TRAINING_CURVES = "pages/3_2_training_curves.py"
PAGE_GRADCAM_DEMO = "pages/3_4_gradcam_demo.py"

# Fixed 4-image demo folders (ONLY these folders)
DEMO4_FIXED_DIR = STREAMLIT_DIR / "assets" / "demo4_fixed"
DEMO4_COVID_DIR = DEMO4_FIXED_DIR / "covid"
DEMO4_NONCOVID_DIR = DEMO4_FIXED_DIR / "noncovid"

# Model checkpoint (direct path, no app_paths import)
MODEL_PATH = PROJECT_ROOT / "models" / "results" / "resnet50_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Style (consistent with 3.x pages)
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

/* Sample cards */
.sample-card{
    border:1px solid #d9e1ea;
    border-radius:12px;
    background:#ffffff;
    padding:10px 12px;
    margin-top:6px;
}
.sample-meta{
    margin:0 0 4px 0;
    font-size:0.92rem;
}
.sample-meta b{
    color:#111827;
}
.pred-ok{
    color:#16a34a;
    font-weight:700;
}
.pred-bad{
    color:#dc2626;
    font-weight:700;
}
.status-chip{
    display:inline-block;
    padding:2px 8px;
    border-radius:999px;
    font-size:0.78rem;
    font-weight:700;
    margin-bottom:8px;
}
.chip-ok{
    background:#ecfdf3;
    color:#166534;
    border:1px solid #bbf7d0;
}
.chip-bad{
    background:#fef2f2;
    color:#991b1b;
    border:1px solid #fecaca;
}

/* Mono numbers */
.mono{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
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


# =========================================================
# UI helpers
# =========================================================
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


# =========================================================
# Model + inference
# =========================================================
@st.cache_resource
def load_model_and_threshold():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    best_threshold = 0.40

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if "best_threshold" in ckpt:
            try:
                best_threshold = float(ckpt["best_threshold"])
            except Exception:
                best_threshold = 0.40
    elif isinstance(ckpt, dict):
        # plain state dict
        model.load_state_dict(ckpt)
    else:
        raise RuntimeError("Checkpoint format not recognized.")

    model.to(DEVICE).eval()
    return model, best_threshold


@st.cache_resource
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def predict_covid_prob(model, tfm, img: Image.Image) -> float:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[1])  # class1 = covid


# =========================================================
# Fixed demo set loading (ONLY demo4_fixed folders)
# =========================================================
def _list_image_files(folder: Path):
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
    )


def _folder_signature(folder: Path):
    """Used to auto-refresh cache when files change."""
    files = _list_image_files(folder)
    sig = []
    for p in files:
        try:
            sig.append((p.name, p.stat().st_mtime_ns, p.stat().st_size))
        except Exception:
            sig.append((p.name, 0, 0))
    return tuple(sig)


@st.cache_resource
def load_demo_records_cached(covid_sig, noncovid_sig, model_sig):
    """
    Build + cache fixed 4-image demo (2 COVID + 2 Non-COVID) and precompute probabilities once.
    Threshold slider then only changes labels (fast).
    """
    model, default_threshold = load_model_and_threshold()
    tfm = get_transform()

    covid_files = _list_image_files(DEMO4_COVID_DIR)[:2]
    noncovid_files = _list_image_files(DEMO4_NONCOVID_DIR)[:2]

    if len(covid_files) < 2 or len(noncovid_files) < 2:
        return [], default_threshold

    records = []

    for true_cls, files in [("covid", covid_files), ("noncovid", noncovid_files)]:
        for p in files:
            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
                    prob = predict_covid_prob(model, tfm, img)
                    records.append(
                        {
                            "true_class": true_cls,
                            "filename": p.name,
                            "image": img.copy(),  # keep image in cache
                            "p_covid": prob,  # precomputed once
                        }
                    )
            except Exception:
                # skip unreadable files
                continue

    # stable order: covid first, then noncovid
    records.sort(key=lambda r: (0 if r["true_class"] == "covid" else 1, r["filename"]))
    return records, default_threshold


# =========================================================
# Header
# =========================================================
st.title("Inference Demo — ResNet50 (Borderline Samples)")
st.caption(
    "Interactive threshold demo on 4 fixed borderline X-ray samples (2 COVID, 2 non-COVID). "
    "Educational demonstration only — not for clinical diagnosis."
)

# =========================================================
# Top navigation
# =========================================================
n1, n2 = st.columns(2)
with n1:
    if st.button("⬅ Training Curves", use_container_width=True, key="nav_top_train"):
        safe_switch(PAGE_TRAINING_CURVES)

with n2:
    if st.button("Grad-CAM Demo ➜", use_container_width=True, key="nav_top_gradcam"):
        safe_switch(PAGE_GRADCAM_DEMO)

st.markdown("---")

# =========================================================
# Guards
# =========================================================
if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

# Build signatures so cache updates automatically if files are replaced
covid_sig = _folder_signature(DEMO4_COVID_DIR)
noncovid_sig = _folder_signature(DEMO4_NONCOVID_DIR)
try:
    model_sig = (
        MODEL_PATH.name,
        MODEL_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_size,
    )
except Exception:
    model_sig = ("missing", 0, 0)

records_all, default_threshold = load_demo_records_cached(
    covid_sig, noncovid_sig, model_sig
)

if len(records_all) < 4:
    st.warning(
        "Fixed demo set not found (need 2 images in each folder):\n\n"
        "- `src/streamlit/assets/demo4_fixed/covid/`\n"
        "- `src/streamlit/assets/demo4_fixed/noncovid/`\n\n"
        "Please place 2 actual X-ray images per class (no masks)."
    )
    with st.expander("Paths / debug"):
        st.write("MODEL_PATH:", str(MODEL_PATH), "| exists:", MODEL_PATH.exists())
        st.write(
            "DEMO4_COVID_DIR:",
            str(DEMO4_COVID_DIR),
            "| exists:",
            DEMO4_COVID_DIR.exists(),
        )
        st.write(
            "DEMO4_NONCOVID_DIR:",
            str(DEMO4_NONCOVID_DIR),
            "| exists:",
            DEMO4_NONCOVID_DIR.exists(),
        )
        st.write("COVID files:", [p.name for p in _list_image_files(DEMO4_COVID_DIR)])
        st.write(
            "Non-COVID files:", [p.name for p in _list_image_files(DEMO4_NONCOVID_DIR)]
        )
    st.stop()

# =========================================================
# Controls
# =========================================================
st.subheader("Interactive Threshold Demo")

c1, c2 = st.columns([2, 1], gap="large")
with c1:
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, float(default_threshold), 0.01
    )
with c2:
    class_filter = st.selectbox("Class filter", ["all", "covid", "noncovid"], index=0)

# =========================================================
# Filter + live decisions (fast: probabilities already cached)
# =========================================================
records = records_all
if class_filter != "all":
    records = [r for r in records_all if r["true_class"] == class_filter]

pred_rows = []
for r in records:
    pred = "covid" if float(r["p_covid"]) >= threshold else "noncovid"
    pred_rows.append((r["true_class"], pred))

n_total = len(pred_rows)
n_correct = sum(1 for t, p in pred_rows if t == p)
n_wrong = n_total - n_correct

# =========================================================
# Policy + demo notes
# =========================================================
card(
    f"""
    <b>Live policy note:</b> This demo uses a <b>recall-first screening perspective</b> (focus on reducing missed positives / FN).<br>
    <b>Operating threshold shown live:</b> <span class="mono">{threshold:.2f}</span>
    """,
    cls="info-card",
)

card(
    """
    <b>Demo note:</b> The counts below are based on the <b>currently shown fixed samples</b> (demo view), not on the full test set.
    Use this page to illustrate how changing the threshold can flip predictions on borderline cases.
    """,
    cls="soft-card",
)

# =========================================================
# KPIs (demo-view counts)
# =========================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Samples shown", str(n_total), "current filtered view")
with k2:
    kpi("Threshold", f"{threshold:.2f}", "live decision cut-off")
with k3:
    kpi("Matches", str(n_correct), "prediction == true label")
with k4:
    kpi("Mismatches", str(n_wrong), "prediction != true label")

st.markdown("")

# =========================================================
# Image grid (4 columns) — same functionality, cleaner presentation
# =========================================================
if not records:
    st.info("No samples match the selected class filter.")
else:
    cols = st.columns(4, gap="large")

    for i, r in enumerate(records):
        with cols[i % 4]:
            img = r["image"]
            true_cls = r["true_class"]
            prob = float(r["p_covid"])
            pred = "covid" if prob >= threshold else "noncovid"
            ok = pred == true_cls

            st.image(img, use_column_width=True, caption=r["filename"])

            chip_html = (
                "<span class='status-chip chip-ok'>Match</span>"
                if ok
                else "<span class='status-chip chip-bad'>Mismatch</span>"
            )
            pred_html = (
                f"<span class='pred-ok'>✅ {pred}</span>"
                if ok
                else f"<span class='pred-bad'>❌ {pred}</span>"
            )

            st.markdown(
                f"""
<div class="sample-card">
  {chip_html}
  <div class="sample-meta"><b>True label:</b> <span class="mono">{true_cls}</span></div>
  <div class="sample-meta"><b>Live p(covid):</b> <span class="mono">{prob:.3f}</span></div>
  <div class="sample-meta" style="margin-bottom:0;"><b>Prediction:</b> {pred_html}</div>
</div>
""",
                unsafe_allow_html=True,
            )

st.divider()

# =========================================================
# Bottom navigation (same buttons, unique keys)
# =========================================================
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("⬅ Training Curves", use_container_width=True, key="nav_bottom_train"):
        safe_switch(PAGE_TRAINING_CURVES)

with b3:
    if st.button("Grad-CAM Demo ➜", use_container_width=True, key="nav_bottom_gradcam"):
        safe_switch(PAGE_GRADCAM_DEMO)
