from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="COVID-19 X-ray Feature Pipeline", layout="wide")

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
      .hero { padding: 18px 6px 10px 6px; }
      .hero-title { font-size: 56px; font-weight: 900; letter-spacing: -0.03em; line-height: 1.02; margin: 0; }
      .hero-sub { font-size: 18px; opacity: 0.84; margin-top: 12px; max-width: 980px; }
      .pill { display:inline-block; padding: 6px 12px; border-radius: 999px;
              background: rgba(49,51,63,0.08); margin-right: 8px; margin-top: 10px;
              font-size: 12.5px; }
      .divider { height: 1px; background: rgba(49,51,63,0.12); margin: 16px 0; }
      .kpi { padding: 14px 14px; border-radius: 18px; background: rgba(49,51,63,0.05);
             border: 1px solid rgba(49,51,63,0.10); }
      .kpi-title { font-weight: 800; font-size: 13px; opacity: 0.8; }
      .kpi-value { font-weight: 900; font-size: 30px; letter-spacing: -0.01em; margin-top: 2px; }
      .kpi-sub { font-size: 12.5px; opacity: 0.75; margin-top: 2px; }
      .section-title { font-size: 28px; font-weight: 900; letter-spacing: -0.01em; margin: 4px 0 6px 0; }
      .section-sub { font-size: 14px; opacity: 0.82; margin: 0 0 10px 0; max-width: 1050px; }
      .label { font-size: 13px; font-weight: 700; opacity: 0.85; margin: 0 0 6px 0; }
      .note { font-size: 12.5px; opacity: 0.70; }
      .card { padding: 14px; border-radius: 18px; background: rgba(49,51,63,0.05);
              border: 1px solid rgba(49,51,63,0.10); }
      div.stButton > button {
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PATHS (FIXED)
# __file__ = .../src/streamlit/pages/1_eda_feature_engineering.py
# repo root = parents[2]
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent  # .../src/streamlit/pages
BASE_DIR = CURRENT_DIR.parents[2]  # .../nov25_bds_int_covid1 (repo root)

# Preferred location in this repo
DATA_DIR = BASE_DIR / "data" / "processed" / "ML"

# Optional fallback for older branches/layouts
ALT_DATA_DIR = BASE_DIR / "src" / "data" / "processed" / "ML"

FEATURES_ALL_PATH = DATA_DIR / "features_reduced_corr.csv"
FEATURES_15_PATH = DATA_DIR / "features_reduced_15.csv"

if (not FEATURES_ALL_PATH.exists() or not FEATURES_15_PATH.exists()) and (
    (ALT_DATA_DIR / "features_reduced_corr.csv").exists()
    and (ALT_DATA_DIR / "features_reduced_15.csv").exists()
):
    DATA_DIR = ALT_DATA_DIR
    FEATURES_ALL_PATH = DATA_DIR / "features_reduced_corr.csv"
    FEATURES_15_PATH = DATA_DIR / "features_reduced_15.csv"

SAMPLES_DIR = BASE_DIR / "data" / "samples"

XRAY_IMG = CURRENT_DIR / "COVID-32-xray.png"
MASK_IMG = CURRENT_DIR / "COVID-32-mask.png"
MASKED_IMG = CURRENT_DIR / "COVID-32-isolated.png"

# =========================================================
# NAVIGATION
# =========================================================
PAGES = [
    ("intro", "Intro"),
    ("dataset", "Dataset overview"),
    ("masking", "Masking: focus on lungs"),
    ("features", "Feature extraction & categories"),
    ("reduce", "Reduce redundancy → 15"),
    ("ml", "Hand-off to ML"),
]


def set_page(key: str):
    st.session_state["page"] = key


def get_page() -> str:
    if "page" not in st.session_state:
        st.session_state["page"] = "intro"
    return st.session_state["page"]


def clickable_flow(raw_ok: bool, masks_ok: bool, feats_ok: bool):
    steps = [
        ("intro", "Intro"),
        ("dataset", f"Raw X-rays{' ✅' if raw_ok else ''}"),
        ("masking", f"Lung masks{' ✅' if masks_ok else ''}"),
        ("features", f"Features{' ✅' if feats_ok else ''}"),
        ("reduce", "Reduce → 15"),
        ("ml", "ML hand-off"),
    ]

    cols = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1, 0.25, 1, 0.25, 1])
    btn_cols = [0, 2, 4, 6, 8, 10]
    current = get_page()

    for (key, label), col_idx in zip(steps, btn_cols):
        with cols[col_idx]:
            shown = f"● {label}" if key == current else label
            if st.button(shown, key=f"nav_{key}", use_container_width=True):
                set_page(key)
                st.rerun()

    for arrow_idx in [1, 3, 5, 7, 9]:
        with cols[arrow_idx]:
            st.markdown(
                "<div style='text-align:center; opacity:0.55; font-size:18px;'>→</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='note'>Click any step to navigate.</div>", unsafe_allow_html=True
    )


# =========================================================
# HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_image(path):
    return Image.open(path)


def safe_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str)
    return out


def numeric_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["target"], errors="ignore").select_dtypes(
        include=[np.number]
    )


def find_example_images(root_dir, class_folder: str, max_n=2) -> List[str]:
    """
    Returns sample image file paths from a class folder.
    Uses repo-contained samples for portability.
    """
    root_dir = os.fspath(root_dir)
    if not root_dir or not os.path.isdir(root_dir):
        return []

    patterns = [
        os.path.join(root_dir, class_folder, "images", "*.png"),
        os.path.join(root_dir, class_folder, "images", "*.jpg"),
        os.path.join(root_dir, class_folder, "*.png"),
        os.path.join(root_dir, class_folder, "*.jpg"),
        os.path.join(root_dir, class_folder, "**", "*.png"),
        os.path.join(root_dir, class_folder, "**", "*.jpg"),
    ]
    candidates: List[str] = []
    for p in patterns:
        candidates.extend(glob.glob(p, recursive=True))

    candidates = sorted(list(set(candidates)))
    return candidates[:max_n]


# =========================================================
# PLOTS (compact, presentation-friendly; NO custom theme)
# =========================================================
def plot_class_balance(df: pd.DataFrame):
    counts = df["target"].value_counts().sort_index()
    labels = ["Non-COVID (0)", "COVID (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(2.6, 1.6), dpi=170)
    ax.bar(labels, values)

    ax.set_title("Class balance", fontsize=9, pad=4)
    ax.set_ylabel("Count", fontsize=8)

    ax.tick_params(axis="x", length=0, labelrotation=0, labelsize=7)
    ax.tick_params(axis="y", length=0, labelsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.6)
    return fig


def corr_filter(X: pd.DataFrame, threshold: float):
    corr = X.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c].abs() >= threshold)]
    X_red = X.drop(columns=to_drop, errors="ignore")
    return X_red, to_drop, corr


def short_label(s: str, maxlen: int = 18) -> str:
    s = str(s)
    return s if len(s) <= maxlen else (s[: maxlen - 3] + "...")


def heatmap_with_labels(corr: pd.DataFrame, cols: list, title: str):
    sub = corr.loc[cols, cols]
    n = len(cols)

    fig_w = min(8.0, max(5.5, 0.32 * n))
    fig_h = min(6.0, max(3.8, 0.28 * n))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    ax.grid(False)

    im = ax.imshow(
        sub.values,
        aspect="auto",
        interpolation="none",
        cmap="Blues",
        vmin=-1.0,
        vmax=1.0,
    )

    ax.set_title(title, fontsize=9, pad=6)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        [short_label(c) for c in cols], rotation=60, ha="right", fontsize=6
    )
    ax.set_yticklabels([short_label(c) for c in cols], fontsize=6)

    ax.tick_params(axis="both", length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.8)
    return fig


def flowchart_graphviz():
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR;
          node [shape=box, style="rounded,filled", color="#888888", fillcolor="#F2F2F2", fontname="Arial"];
          A [label="Raw X-rays\\n(COVID / Normal / Viral Pneumonia / Lung Opacity)"];
          B [label="Binary mapping\\nCOVID vs Non-COVID"];
          C [label="Lung mask"];
          D [label="Masked lung region"];
          E [label="Multivariate features\\n(intensity, texture, freq, edges, shape)"];
          F [label="Asymmetry\\n(L vs R)"];
          G [label="Multicollinearity control\\n(correlation reduction)"];
          H [label="Final 15 features\\nstable + interpretable"];
          I [label="ML modeling"];
          A -> B -> C -> D -> E -> F -> G -> H -> I;
        }
        """
    )


def categorize_feature(name: str) -> str:
    n = str(name).lower()

    if n in {"image_name", "label", "target"} or "name" in n:
        return "Metadata / Labels"
    if "size" in n or "kb" in n:
        return "Metadata / Labels"
    if any(k in n for k in ["mean", "std", "var", "skew", "kurt"]):
        return "Statistical Intensity"
    if any(
        k in n
        for k in ["glcm", "contrast", "homogeneity", "energy", "correlation", "fft"]
    ):
        return "Texture (GLCM / Frequency)"
    if any(k in n for k in ["laplacian", "gradient", "grad_mag", "edge"]):
        return "Edge / Sharpness"
    if "lbp" in n:
        return "Local Binary Pattern"
    if any(k in n for k in ["area", "perimeter", "ratio", "bbox"]):
        return "Shape / Morphology"
    if any(k in n for k in ["opacity_compactness", "opacity_eccentricity", "opacity"]):
        return "Opacity Geometry"
    return "Other"


def category_counts(feature_names: List[str]) -> pd.DataFrame:
    cat = pd.Series([categorize_feature(f) for f in feature_names]).value_counts()
    out = cat.reset_index()
    out.columns = ["Category", "Count"]
    return out


def plot_category_counts(df_counts: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=170)

    y = np.arange(len(df_counts))
    ax.barh(y, df_counts["Count"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(df_counts["Category"].values, fontsize=7)
    ax.invert_yaxis()

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("Count", fontsize=8)

    ax.tick_params(axis="x", length=0, labelsize=7)
    ax.tick_params(axis="y", length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.6)
    return fig


def median_abs_corr(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return 0.0
    vals = corr.values
    mask = ~np.eye(vals.shape[0], dtype=bool)
    off = vals[mask]
    off = off[~np.isnan(off)]
    if off.size == 0:
        return 0.0
    return float(np.median(np.abs(off)))


def max_abs_corr(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return 0.0
    vals = corr.values
    mask = ~np.eye(vals.shape[0], dtype=bool)
    off = vals[mask]
    off = off[~np.isnan(off)]
    if off.size == 0:
        return 0.0
    return float(np.max(np.abs(off)))


# =========================================================
# LOAD DATA
# =========================================================
missing = []
if not FEATURES_ALL_PATH.exists():
    missing.append(FEATURES_ALL_PATH)
if not FEATURES_15_PATH.exists():
    missing.append(FEATURES_15_PATH)

if missing:
    st.error("Missing required CSV files. Check these paths:")
    for p in missing:
        st.write(f"- `{p}`")
    with st.expander("Debug paths", expanded=True):
        st.write(f"CURRENT_DIR: `{CURRENT_DIR}`")
        st.write(f"BASE_DIR: `{BASE_DIR}`")
        st.write(f"DATA_DIR (active): `{DATA_DIR}`")
        st.write(f"ALT_DATA_DIR: `{ALT_DATA_DIR}`")
    st.stop()

df_all = load_csv(FEATURES_ALL_PATH)
df_15 = load_csv(FEATURES_15_PATH)

if "target" not in df_all.columns or "target" not in df_15.columns:
    st.error("Both CSVs must contain a 'target' column (0=Non-COVID, 1=COVID).")
    st.stop()

X_all = numeric_X(df_all)
X_15 = numeric_X(df_15)

df_all_disp = safe_df_for_streamlit(df_all)
df_15_disp = safe_df_for_streamlit(df_15)

# cache demo images (prevents jitter)
xray_img = load_image(XRAY_IMG) if XRAY_IMG.exists() else None
mask_img = load_image(MASK_IMG) if MASK_IMG.exists() else None
masked_img = load_image(MASKED_IMG) if MASKED_IMG.exists() else None

raw_dir = SAMPLES_DIR

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("### Settings")
corr_threshold = st.sidebar.slider("Correlation threshold", 0.70, 0.99, 0.90, 0.01)
heatmap_n = st.sidebar.slider("Heatmap features shown", 10, 30, 18, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Raw samples path (repo)")
st.sidebar.code(str(raw_dir))

st.sidebar.markdown("---")
st.sidebar.markdown("### Jump to")
for key, label in PAGES:
    if st.sidebar.button(label, key=f"sb_{key}", use_container_width=True):
        set_page(key)
        st.rerun()

# =========================================================
# TOP CLICKABLE FLOW
# =========================================================
raw_ok = raw_dir.is_dir()
masks_ok = (
    (xray_img is not None) and (mask_img is not None) and (masked_img is not None)
)
feats_ok = (X_all.shape[1] > 0) and (X_15.shape[1] > 0)

clickable_flow(raw_ok=raw_ok, masks_ok=masks_ok, feats_ok=feats_ok)

with st.expander("Pipeline overview (flowchart)", expanded=False):
    flowchart_graphviz()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

page = get_page()

# Quantitative indicators (start → final 15)
start_features = int(X_all.shape[1])
final_features = int(X_15.shape[1])
removed_pct = 100.0 * (start_features - final_features) / max(1, start_features)
kept_pct = 100.0 * final_features / max(1, start_features)

# =========================================================
# PAGE: INTRO
# =========================================================
if page == "intro":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Identification of COVID-19 cases<br>from chest X-rays</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">'
        "The pipeline isolates lung regions, extracts a multivariate feature set from multiple radiomic families, "
        "and then controls multicollinearity to obtain a compact, stable feature table for Machine Learning."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="pill">Multivariate signal</span>'
        '<span class="pill">Multicollinearity control</span>'
        '<span class="pill">Mask-focused lung ROI</span>'
        '<span class="pill">Generalization-first</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">Samples</div>'
            f'<div class="kpi-value">{df_15.shape[0]:,}</div>'
            f'<div class="kpi-sub">rows in final dataset</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">Features (start)</div>'
            f'<div class="kpi-value">{start_features}</div>'
            f'<div class="kpi-sub">multivariate descriptors</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">Final features</div>'
            f'<div class="kpi-value">{final_features}</div>'
            f'<div class="kpi-sub">compact ML-ready set</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">Removed</div>'
            f'<div class="kpi-value">{removed_pct:.1f}%</div>'
            f'<div class="kpi-sub">redundancy reduction</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">Kept</div>'
            f'<div class="kpi-value">{kept_pct:.1f}%</div>'
            f'<div class="kpi-sub">informative subset</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    c_plot, c_space = st.columns([0.42, 0.58])
    with c_plot:
        st.pyplot(plot_class_balance(df_15), width="stretch")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
**Multivariate vs multicollinearity**

- **Multivariate signal**: COVID-related changes are not captured by a single metric. Complementary feature families
  (intensity, texture, edges, frequency, geometry) describe lung appearance from different viewpoints.
- **Multicollinearity**: engineered predictors can be strongly correlated (left/right variants, asymmetry ratios, derived measures).
  Without control, models become less stable and overfit.
- **Approach**: preserve multivariate information, remove redundancy, and use a compact feature set for ML.
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE: DATASET OVERVIEW
# =========================================================
elif page == "dataset":
    st.markdown(
        '<div class="section-title">Raw dataset → binary target</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "For the Streamlit demo, we ship a small set of representative samples inside the repo "
        "(data/samples). Training and feature extraction were performed on the full dataset offline."
        "</div>",
        unsafe_allow_html=True,
    )

    if not raw_dir.is_dir():
        st.error(f"Samples folder not found: {raw_dir}")
        st.info(
            "Expected structure: data/samples/COVID, Normal, Viral_Pneumonia, Lung_Opacity"
        )
        st.stop()

    # display name vs folder name (so your folder names can contain underscores)
    raw_classes: List[Tuple[str, str]] = [
        ("COVID", "COVID"),
        ("Normal", "Normal"),
        ("Viral Pneumonia", "Viral_Pneumonia"),
        ("Lung Opacity", "Lung_Opacity"),
    ]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="label">Raw categories (sample images from repo)</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for i, (display_name, folder_name) in enumerate(raw_classes):
        with cols[i]:
            st.write(display_name)
            examples = find_example_images(raw_dir, folder_name, max_n=2)
            if examples:
                for p in examples:
                    st.image(load_image(p), width=210)
            else:
                st.caption(f"No sample images found in: data/samples/{folder_name}/")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    mapping_df = pd.DataFrame(
        {
            "Original label": ["COVID", "Normal", "Viral Pneumonia", "Lung Opacity"],
            "Final label": [
                "COVID (1)",
                "Non-COVID (0)",
                "Non-COVID (0)",
                "Non-COVID (0)",
            ],
        }
    )
    st.markdown('<div class="label">Binary mapping used</div>', unsafe_allow_html=True)
    st.dataframe(mapping_df, width="stretch", height=180)

# =========================================================
# PAGE: MASKING
# =========================================================
elif page == "masking":
    st.markdown(
        '<div class="section-title">Masking: isolate the lung region</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "Lung masks isolate the region of interest so extracted features describe lung tissue only."
        "</div>",
        unsafe_allow_html=True,
    )

    FIXED_W = 330
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.markdown('<div class="label">Original X-ray</div>', unsafe_allow_html=True)
        if xray_img:
            st.image(xray_img, width=FIXED_W)
        else:
            st.warning("Missing: COVID-32-xray.png")

    with col2:
        st.markdown('<div class="label">Lung mask</div>', unsafe_allow_html=True)
        if mask_img:
            st.image(mask_img, width=FIXED_W)
        else:
            st.warning("Missing: COVID-32-mask.png")

    with col3:
        st.markdown(
            '<div class="label">Masked lung region</div>', unsafe_allow_html=True
        )
        if masked_img:
            st.image(masked_img, width=FIXED_W)
        else:
            st.warning("Missing: COVID-32-isolated.png")

# =========================================================
# PAGE: FEATURE EXTRACTION
# =========================================================
elif page == "features":
    st.markdown(
        '<div class="section-title">Feature extraction: multivariate representation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "After lung isolation, multiple feature families are extracted to capture complementary patterns. "
        "This multivariate representation improves separability for ML."
        "</div>",
        unsafe_allow_html=True,
    )

    all_features = [c for c in df_all.columns if c != "target"]
    f15_features = [c for c in df_15.columns if c != "target"]

    all_counts = category_counts(all_features)
    f15_counts = category_counts(f15_features)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            '<div class="label">Categories in features_reduced_corr.csv</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(all_counts, width="stretch", height=220)
        st.pyplot(
            plot_category_counts(all_counts, "Category distribution (corr-reduced)"),
            width="stretch",
        )

    with c2:
        st.markdown(
            '<div class="label">Categories in features_reduced_15.csv</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(f15_counts, width="stretch", height=220)
        st.pyplot(
            plot_category_counts(f15_counts, "Category distribution (final 15)"),
            width="stretch",
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
**Multivariate feature families**
- **Intensity statistics**: brightness distribution and heterogeneity.
- **Texture (GLCM / frequency)**: patchiness and structural patterns.
- **Edge / sharpness**: boundary strength and blur-related changes.
- **Shape / morphology** and **asymmetry**: spatial distribution and left–right differences.

The final ML set preserves this multivariate signal while removing redundant predictors.
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="label">Feature table preview (corr-reduced)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df_all_disp.head(18), width="stretch", height=300)

# =========================================================
# PAGE: REDUCE TO 15
# =========================================================
elif page == "reduce":
    st.markdown(
        '<div class="section-title">Multicollinearity control: reduce redundancy → final 15</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "Multivariate features are valuable, but many engineered predictors are correlated. "
        "Reducing multicollinearity improves stability and generalization."
        "</div>",
        unsafe_allow_html=True,
    )

    X_red, to_drop, corr = corr_filter(X_all, corr_threshold)
    corr_after = X_red.corr() if X_red.shape[1] >= 2 else pd.DataFrame(np.eye(1))

    before_med = median_abs_corr(corr)
    after_med = median_abs_corr(corr_after)
    before_max = max_abs_corr(corr)
    after_max = max_abs_corr(corr_after)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        f'<div class="kpi"><div class="kpi-title">Start</div>'
        f'<div class="kpi-value">{X_all.shape[1]}</div><div class="kpi-sub">numeric features</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="kpi"><div class="kpi-title">Dropped</div>'
        f'<div class="kpi-value">{len(to_drop)}</div><div class="kpi-sub">|corr| ≥ {corr_threshold:.2f}</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="kpi"><div class="kpi-title">Remaining</div>'
        f'<div class="kpi-value">{X_red.shape[1]}</div><div class="kpi-sub">after filtering</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="kpi"><div class="kpi-title">Median |corr|</div>'
        f'<div class="kpi-value">{before_med:.2f} → {after_med:.2f}</div><div class="kpi-sub">redundancy drop</div></div>',
        unsafe_allow_html=True,
    )
    c5.markdown(
        f'<div class="kpi"><div class="kpi-title">Max |corr|</div>'
        f'<div class="kpi-value">{before_max:.2f} → {after_max:.2f}</div><div class="kpi-sub">worst-case drop</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    var_rank = X_all.var().sort_values(ascending=False).index.tolist()
    cols = var_rank[:heatmap_n]
    fig = heatmap_with_labels(
        corr, cols, title=f"Correlation heatmap (top {heatmap_n} by variance)"
    )
    st.pyplot(fig, width="stretch")

    corr_features = [c for c in df_all.columns if c != "target"]
    final_features_list = [c for c in df_15.columns if c != "target"]
    removed = sorted(list(set(corr_features) - set(final_features_list)))
    common = sorted(list(set(corr_features).intersection(set(final_features_list))))

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total features (corr-reduced file)",
                "Final selected features (15 set)",
                "Removed after final selection",
                "Common features retained",
            ],
            "Count": [
                len(corr_features),
                len(final_features_list),
                len(removed),
                len(common),
            ],
        }
    )

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown(
            '<div class="label">Set comparison summary</div>', unsafe_allow_html=True
        )
        st.dataframe(summary, width="stretch", height=180)
        st.markdown(
            '<div class="label">Removed features (preview)</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame({"Removed": removed[:35]}), width="stretch", height=300
        )

    with cB:
        st.markdown(
            '<div class="label">Final 15 features (used for ML)</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame({"Final 15": final_features_list}), width="stretch", height=360
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
**Why multicollinearity reduction is essential for ML**

- Many radiomic predictors encode overlapping information (left/right versions + asymmetry + derived ratios).
- High multicollinearity can cause unstable learning and overfitting.
- The final set preserves multivariate information while removing redundant predictors for robust ML behavior.
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE: ML HAND-OFF
# =========================================================
elif page == "ml":
    st.markdown(
        '<div class="section-title">ML-ready dataset: stable multivariate input</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "The final table preserves multivariate information while controlling multicollinearity. "
        "This provides a stable foundation for ML classification."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.markdown(
            '<div class="label">Exported datasets</div>', unsafe_allow_html=True
        )
        export_df = pd.DataFrame(
            {
                "Dataset": ["features_reduced_corr.csv", "features_reduced_15.csv"],
                "Meaning": [
                    "multivariate feature set after initial redundancy reduction",
                    "final compact set with controlled multicollinearity",
                ],
                "Path": [str(FEATURES_ALL_PATH), str(FEATURES_15_PATH)],
            }
        )
        st.dataframe(export_df, width="stretch", height=150)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="label">Why this is best for ML</div>', unsafe_allow_html=True
        )
        st.write("• preserves multivariate representation across feature families")
        st.write("• reduces multicollinearity → stable learning behavior")
        st.write("• lowers overfitting risk → better generalization")
        st.write("• improves interpretability → easier clinical justification")

    with c2:
        st.markdown(
            '<div class="label">Final dataset preview (15 + target)</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df_15_disp.head(22), width="stretch", height=420)
