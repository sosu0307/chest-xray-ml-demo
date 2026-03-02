# ==========================================
# Distance-weighted KNN for COVID vs Non-COVID
# k restricted to 3–5
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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
y = df["target"]

if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Classes:", list(le.classes_))
else:
    print("Target already numeric")

# assume COVID = 1
pos_label = 1

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
    random_state=42,
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
        weights="distance"   # KEY CHANGE
    )

    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1  = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)

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
print(classification_report(y_test, best_pred, zero_division=0))
