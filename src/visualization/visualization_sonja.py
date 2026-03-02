# ============================================================
# PLOT A: Feature Distributions (LEFT vs RIGHT vs ASYM)
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\ML\features_reduced_corr.csv"
)

FEATURE = "entropy"  # change to other base features

cols = [
    f"{FEATURE}_lunge-left",
    f"{FEATURE}_lunge-right",
    f"{FEATURE}_asym_relative",
]

df_plot = df[cols].melt(var_name="feature", value_name="value")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="feature", y="value")
plt.title(f"Distribution of {FEATURE}: Left vs Right vs Asymmetry")
plt.tight_layout()
plt.show()


# ============================================================
# PLOT B: Left vs Right Correlation
# ============================================================

import numpy as np

FEATURE = "entropy"

x = df[f"{FEATURE}_lunge-left"]
y = df[f"{FEATURE}_lunge-right"]

corr = x.corr(y)

plt.figure(figsize=(5, 5))
plt.scatter(x, y, alpha=0.2, s=5)
plt.xlabel("Left lung")
plt.ylabel("Right lung")
plt.title(f"{FEATURE}: Left vs Right (r = {corr:.2f})")
plt.tight_layout()
plt.show()


# ============================================================
# PLOT C: Asymmetry vs Mean of Left & Right
# ============================================================

mean_lr = (x + y) / 2
asym = df[f"{FEATURE}_asym_relative"]

corr_asym = asym.corr(mean_lr)

plt.figure(figsize=(5, 5))
plt.scatter(mean_lr, asym, alpha=0.2, s=5)
plt.xlabel("Mean(Left, Right)")
plt.ylabel("Asymmetry")
plt.title(f"Asym vs Mean (r = {corr_asym:.2f})")
plt.tight_layout()
plt.show()

# ============================================================
# PLOT D: Feature Composition by Type
# ============================================================

feature_types = (
    df.columns
      .to_series()
      .str.extract(r'_(lunge-left|lunge-right|asym_relative)')
      .dropna()[0]
)

feature_types.value_counts().plot(
    kind="bar",
    figsize=(6, 4),
    title="Feature Composition by Lung Representation"
)

plt.ylabel("Number of features")
plt.tight_layout()
plt.show()


