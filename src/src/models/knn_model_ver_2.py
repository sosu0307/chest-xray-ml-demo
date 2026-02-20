

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
POS_LABEL = 1  # COVID = 1

# -------------------------
# STEP 1: Load data
# -------------------------
DATA_PATH = Path(
    r"C:\Projects\GitHub\nov25_bds_int_covid1\data\processed\ML\features_reduced_15.csv"
)
df = pd.read_csv(DATA_PATH)

print("Loaded:", df.shape)

# -------------------------
# STEP 2: Target
# -------------------------
# We make sure:
#   COVID -> 1
#   Non-COVID (and anything else) -> 0
y_raw = df["target"]

if y_raw.dtype == "object":
    y_clean = y_raw.astype(str).str.strip()

    # Mapping (extend if your dataset has other label spellings)
    mapping = {
        "COVID": 1,
        "COVID-19": 1,
        "covid": 1,
        "Covid": 1,
        "Non-COVID": 0,
        "NON-COVID": 0,
        "non_covid": 0,
        "NonCovid": 0,
        "Normal": 0,
        "NORMAL": 0,
    }

    y = y_clean.map(mapping)

    # If there are unmapped labels, show them clearly and stop
    if y.isna().any():
        unknown = sorted(y_clean[y.isna()].unique().tolist())
        print("\nERROR: Found target labels not covered by mapping:", unknown)
        print("Please add them to the mapping dictionary.")
        raise ValueError("Unmapped target labels present.")

    y = y.astype(int)
    print("Target mapped successfully. Unique values:", sorted(y.unique().tolist()))

else:
    # Already numeric
    y = y_raw.astype(int)
    print("Target already numeric. Unique values:", sorted(pd.Series(y).unique().tolist()))

# -------------------------
# STEP 3: Features
# -------------------------
drop_cols = ["target", "image_label"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
X = X.select_dtypes(include=[np.number])

print("X shape:", X.shape)

# -------------------------
# STEP 4: Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# -------------------------
# STEP 5: Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# STEP 6: Distance-weighted KNN (k = 3–5)
# -------------------------
k_values = [3, 4, 5]
results = []

for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="minkowski",   # Euclidean
        p=2,
        weights="distance"
    )

    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)

    results.append((k, acc, rec, f1))

# -------------------------
# STEP 7: Results
# -------------------------
print("\n=== Distance-weighted KNN results (Euclidean) ===")
print("k | accuracy | covid_recall | covid_f1")
print("----------------------------------------------")
for k, acc, rec, f1 in results:
    print(f"{k:2d} | {acc:.4f}   | {rec:.4f}       | {f1:.4f}")

# -------------------------
# STEP 8: Best model details
# -------------------------
best_k, _, _, _ = max(results, key=lambda x: x[3])
print(f"\nBest k by COVID F1 = {best_k}")

best_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="minkowski",
    p=2,
    weights="distance"
)

best_knn.fit(X_train_scaled, y_train)
best_pred = best_knn.predict(X_test_scaled)

print("\nConfusion matrix (best):")
print(confusion_matrix(y_test, best_pred))

print("\nClassification report (best):")
print(
    classification_report(
        y_test,
        best_pred,
        labels=[0, 1],
        target_names=["Non-COVID", "COVID"],
        zero_division=0
    )
)

# -------------------------
# STEP 9: Save model 
# -------------------------
import joblib

MODEL_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\models\knn_model_ver_2.pkl")
SCALER_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\models\scaler_knn.pkl")
FEATURES_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\models\knn_feature_columns.pkl")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save estimator only 
joblib.dump(best_knn, MODEL_PATH)

# Recommended: save scaler and feature columns separately 
joblib.dump(scaler, SCALER_PATH)
joblib.dump(list(X.columns), FEATURES_PATH)

print(f"\nKNN model saved successfully to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
print(f"Feature columns saved to: {FEATURES_PATH}")
