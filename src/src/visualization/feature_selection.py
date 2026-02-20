# ============================================================
# CORRELATION-BASED FEATURE SELECTION
# - Loads merged feature dataset
# - Uses high-correlation pairs
# - Drops redundant features
# - Saves reduced feature dataset
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1) Paths
# ------------------------------------------------------------
DATA_PATH = r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\ML\features_merged_all.csv"
OUT_PATH  = r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\ML\features_reduced_corr.csv"

# ------------------------------------------------------------
# 2) Settings
# ------------------------------------------------------------
CORR_METHOD = "pearson"
HIGH_CORR_THRESHOLD = 0.90

# ------------------------------------------------------------
# 3) Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded merged dataset:", df.shape)

# keep metadata
META_COLS = ["image_name", "label"]
meta = df[META_COLS]

# numeric features only
df_num = df.select_dtypes(include=[np.number])
print("Numeric features:", df_num.shape)

# ------------------------------------------------------------
# 4) Compute correlation matrix
# ------------------------------------------------------------
corr = df_num.corr(method=CORR_METHOD)

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

high_corr = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
)

high_corr = high_corr[high_corr["corr"].abs() >= HIGH_CORR_THRESHOLD]
print("Highly correlated pairs:", len(high_corr))

# ------------------------------------------------------------
# 5) Decide which features to drop
# ------------------------------------------------------------
to_drop = set()

for _, row in high_corr.iterrows():
    f1, f2 = row["feature_1"], row["feature_2"]
    # simple, deterministic rule:
    # keep f1, drop f2
    to_drop.add(f2)

print("Dropping features due to correlation:", len(to_drop))

# ------------------------------------------------------------
# 6) Build reduced dataset
# ------------------------------------------------------------
df_reduced = pd.concat(
    [meta, df_num.drop(columns=list(to_drop), errors="ignore")],
    axis=1
)

print("Reduced dataset shape:", df_reduced.shape)

# ------------------------------------------------------------
# 7) Save result
# ------------------------------------------------------------
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df_reduced.to_csv(OUT_PATH, index=False)

print("Saved reduced feature dataset to:", OUT_PATH)
