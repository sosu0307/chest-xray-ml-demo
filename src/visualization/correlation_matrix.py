# ============================================================
# CORRELATION MATRIX FOR ALL FEATURES (Merged Dataset)
# - Loads merged feature CSV
# - Keeps only numeric feature columns
# - Computes Pearson correlation matrix
# - Plots a correlation heatmap (labels hidden for readability)
# - Lists highly correlated feature pairs (|corr| > threshold)
# - Optionally saves results to CSV
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Paths (EDIT if needed)
# ------------------------------------------------------------
DATA_PATH = r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\ML\features_merged_all.csv"
OUT_DIR = r"D:\DS_ML\Repository\nov25_bds_int_covid1\reprts\figures\correlation_matrix"
OUT_HIGH_CORR_CSV = os.path.join(OUT_DIR, "high_correlations_pairs.csv")

# ------------------------------------------------------------
# 2) Settings
# ------------------------------------------------------------
CORR_METHOD = "pearson"   # "pearson" (default), "spearman" (rank-based)
HIGH_CORR_THRESHOLD = 0.90

# Heatmap size (increase if you want a bigger figure)
FIGSIZE = (18, 14)

# ------------------------------------------------------------
# 3) Load data
# ------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
print("Full dataframe shape:", df.shape)

# ------------------------------------------------------------
# 4) Select numeric feature columns only
# ------------------------------------------------------------
df_num = df.select_dtypes(include=[np.number]).copy()
print("Numeric-only dataframe shape:", df_num.shape)

if df_num.shape[1] < 2:
    raise ValueError("Not enough numeric columns to compute correlations.")

# Optional: drop columns with all NaNs or zero variance (can improve stability)
df_num = df_num.dropna(axis=1, how="all")
zero_var_cols = df_num.columns[df_num.nunique(dropna=True) <= 1].tolist()
if zero_var_cols:
    print(f"Dropping {len(zero_var_cols)} zero-variance columns (example):", zero_var_cols[:10])
    df_num = df_num.drop(columns=zero_var_cols)

print("Numeric dataframe after cleanup shape:", df_num.shape)

# ------------------------------------------------------------
# 5) Compute correlation matrix
# ------------------------------------------------------------
corr = df_num.corr(method=CORR_METHOD)
print("Correlation matrix shape:", corr.shape)

# ------------------------------------------------------------
# 6) Plot correlation heatmap (labels hidden to avoid clutter)
# ------------------------------------------------------------
plt.figure(figsize=FIGSIZE)
plt.imshow(corr.values, aspect="auto", interpolation="nearest")
plt.colorbar(label=f"{CORR_METHOD} correlation")
plt.title(f"Feature Correlation Matrix ({CORR_METHOD.capitalize()})\n(labels hidden for readability)")
plt.xticks([])  # hide labels (too many features)
plt.yticks([])  # hide labels (too many features)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7) Find highly correlated feature pairs (upper triangle)
# ------------------------------------------------------------
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

high_corr = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
)

high_corr["abs_corr"] = high_corr["corr"].abs()
high_corr = high_corr[high_corr["abs_corr"] >= HIGH_CORR_THRESHOLD].sort_values("abs_corr", ascending=False)

print(f"\nHighly correlated feature pairs (|corr| >= {HIGH_CORR_THRESHOLD}):", len(high_corr))
print(high_corr.head(30))

# ------------------------------------------------------------
# 8) Save high-correlation pairs to CSV
# ------------------------------------------------------------
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
high_corr.drop(columns=["abs_corr"]).to_csv(OUT_HIGH_CORR_CSV, index=False)
print("\nSaved high-correlation pairs to:", OUT_HIGH_CORR_CSV)
