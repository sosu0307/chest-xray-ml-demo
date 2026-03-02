# ============================================================
# REPORT-READY PLOTTING SCRIPT
# - Loads reduced feature dataset (CSV)
# - Creates report-friendly plots:
#   (A) Distributions: Left vs Right vs Asymmetry (boxplots)
#   (B) Left vs Right scatter with correlation
#   (C) Asymmetry vs Mean(Left,Right) scatter with correlation
#   (D) Feature composition by type (left/right/asym)
# - Saves all figures to reports/figures/report_plots/
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Paths (EDIT if needed)
# ------------------------------------------------------------
DATA_PATH = r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\ML\features_reduced_corr.csv"
OUT_DIR = Path(r"D:\DS_ML\Repository\nov25_bds_int_covid1\reports\figures\report_plots")

# ------------------------------------------------------------
# 2) Plot settings
# ------------------------------------------------------------
MAX_BASE_FEATURES = 4  # create plots for up to 4 base features
CANDIDATE_BASE_FEATURES = [
    "mean_intensity",
    "rms_contrast",
    "energy",
    "lbp_mean",
    "laplacian_variance",
    "skew",
    "kurtosis",
    "bright_pixel_ratio",
    "dark_pixel_ratio",
]

SCATTER_SAMPLE = 5000     # sample points for scatter for speed/readability
SCATTER_ALPHA = 0.20
SCATTER_SIZE = 6

# ------------------------------------------------------------
# 3) Helpers
# ------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = a.replace([np.inf, -np.inf], np.nan).dropna()
    b = b.replace([np.inf, -np.inf], np.nan).dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        return float("nan")
    return float(a.loc[idx].corr(b.loc[idx]))

def pick_existing_base_features(df_cols) -> list[str]:
    """Pick base feature names that have at least ONE of left/right/asym columns."""
    existing = []
    for base in CANDIDATE_BASE_FEATURES:
        candidates = [
            f"{base}_lunge-left",
            f"{base}_lunge-right",
            f"{base}_asym_relative",
        ]
        if any(c in df_cols for c in candidates):
            existing.append(base)
    return existing[:MAX_BASE_FEATURES]

def melt_for_boxplot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Long format for boxplot; ignores missing columns."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["feature", "value"])
    return df[cols].melt(var_name="feature", value_name="value")

def subsample_for_scatter(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    """Subsample rows for scatter to keep plots readable."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) > n:
        d = d.sample(n=n, random_state=42)
    return d

# ------------------------------------------------------------
# 4) Load data
# ------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
print("Shape:", df.shape)

ensure_dir(OUT_DIR)

# ------------------------------------------------------------
# 5) Plot D: Feature composition by type (left/right/asym)
# ------------------------------------------------------------
col_series = pd.Series(df.columns)
types = col_series.str.extract(r'_(lunge-left|lunge-right|asym_relative)$')[0].dropna()

plt.figure(figsize=(7, 4))
types.value_counts().plot(kind="bar")
plt.title("Feature Composition by Lung Representation")
plt.ylabel("Number of features")
plt.tight_layout()
out_path = OUT_DIR / "D_feature_composition.png"
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)

# ------------------------------------------------------------
# 6) Choose base features that exist in your reduced dataset
# ------------------------------------------------------------
base_features = pick_existing_base_features(df.columns)
print("Base features selected for report plots:", base_features)

if not base_features:
    raise ValueError(
        "No suitable base features found. "
        "Check your column names in features_reduced_corr.csv."
    )

# ------------------------------------------------------------
# 7) For each base feature: A, B, C plots
# ------------------------------------------------------------
for base in base_features:
    left_col = f"{base}_lunge-left"
    right_col = f"{base}_lunge-right"
    asym_col = f"{base}_asym_relative"

    # ---------- Plot A: Distribution (boxplots) ----------
    box_cols = [c for c in [left_col, right_col, asym_col] if c in df.columns]
    df_box = melt_for_boxplot(df, box_cols)

    if not df_box.empty:
        plt.figure(figsize=(9, 5))
        # simple matplotlib boxplot (no seaborn dependency)
        # prepare data in order
        ordered_cols = box_cols
        data_list = [df[c].replace([np.inf, -np.inf], np.nan).dropna().values for c in ordered_cols]
        plt.boxplot(data_list, labels=ordered_cols, showfliers=False)
        plt.xticks(rotation=25, ha="right")
        plt.title(f"A) Distribution: {base} (Left / Right / Asym)")
        plt.ylabel("Value")
        plt.tight_layout()
        out_path = OUT_DIR / f"A_{base}_distribution.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Saved:", out_path)

    # ---------- Plot B: Left vs Right scatter ----------
    if left_col in df.columns and right_col in df.columns:
        d_scatter = subsample_for_scatter(df, [left_col, right_col], SCATTER_SAMPLE)
        if not d_scatter.empty:
            r_lr = safe_corr(d_scatter[left_col], d_scatter[right_col])

            plt.figure(figsize=(5.5, 5.5))
            plt.scatter(d_scatter[left_col], d_scatter[right_col], alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
            plt.xlabel("Left lung")
            plt.ylabel("Right lung")
            plt.title(f"B) {base}: Left vs Right (r = {r_lr:.2f})")
            plt.tight_layout()
            out_path = OUT_DIR / f"B_{base}_left_vs_right.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
            print("Saved:", out_path)

    # ---------- Plot C: Asymmetry vs Mean(Left,Right) ----------
    if left_col in df.columns and right_col in df.columns and asym_col in df.columns:
        d_scatter = subsample_for_scatter(df, [left_col, right_col, asym_col], SCATTER_SAMPLE)
        if not d_scatter.empty:
            mean_lr = (d_scatter[left_col] + d_scatter[right_col]) / 2.0
            r_asym = safe_corr(mean_lr, d_scatter[asym_col])

            plt.figure(figsize=(5.5, 5.5))
            plt.scatter(mean_lr, d_scatter[asym_col], alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
            plt.xlabel("Mean(Left, Right)")
            plt.ylabel("Asymmetry")
            plt.title(f"C) {base}: Asym vs Mean(L,R) (r = {r_asym:.2f})")
            plt.tight_layout()
            out_path = OUT_DIR / f"C_{base}_asym_vs_mean.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
            print("Saved:", out_path)

print("\nDONE. All report plots saved to:")
print(str(OUT_DIR))
