# ==========================================
# KNN: Compare Euclidean vs Manhattan
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
    classification_report,
)

import matplotlib.pyplot as plt


# -------------------------
# STEP 1: Load data
# -------------------------
DATA_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\data\processed\ML\features_reduced_15.csv")
df = pd.read_csv(DATA_PATH)

print("Loaded:", df.shape)
print(df.head())


# -------------------------
# STEP 2: Defining the target
# -------------------------
y = df["target"]

# Convert labels to numbers if needed
le = None
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("\nTarget classes (LabelEncoder order):")
    for i, c in enumerate(le.classes_):
        print(f"  {i} -> {c}")
else:
    print("\nTarget seems numeric already.")

# Decide which label is "COVID" (positive class)
# If your target is already 0/1 with 1=COVID, keep it.
# If it is text, we try to auto-detect which encoded class is COVID.
pos_label = 1
if le is not None:
    classes_lower = [str(c).lower() for c in le.classes_]
    if any("covid" in c for c in classes_lower):
        pos_label = int(np.where([("covid" in c) for c in classes_lower])[0][0])
    else:
        # fallback: assume label "1" is positive
        pos_label = 1

print(f"\nUsing pos_label={pos_label} as the COVID class for recall/F1.\n")


# -------------------------
# STEP 3: Build features
# -------------------------
drop_cols = ["target", "image_label"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
X = X.select_dtypes(include=[np.number])

print("X shape:", X.shape)
print("Feature count:", X.shape[1])


# -------------------------
# STEP 4: Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# STEP 5: Scale features
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# -------------------------
# STEP 6: Compare k + metric
# -------------------------
k_values = [3, 5, 7, 9, 11, 15, 21, 31]

results = {
    "k": [],
    "metric": [],
    "accuracy": [],
    "covid_recall": [],
    "covid_f1": [],
}

def eval_knn(k: int, metric_name: str, p_val: int | None):
    if metric_name == "minkowski":
        model = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=p_val)
    else:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric_name)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1  = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    return acc, rec, f1

# Euclidean = Minkowski p=2
# Manhattan = Minkowski p=1
for k in k_values:
    acc_e, rec_e, f1_e = eval_knn(k, metric_name="minkowski", p_val=2)
    results["k"].append(k)
    results["metric"].append("euclidean")
    results["accuracy"].append(acc_e)
    results["covid_recall"].append(rec_e)
    results["covid_f1"].append(f1_e)

    acc_m, rec_m, f1_m = eval_knn(k, metric_name="minkowski", p_val=1)
    results["k"].append(k)
    results["metric"].append("manhattan")
    results["accuracy"].append(acc_m)
    results["covid_recall"].append(rec_m)
    results["covid_f1"].append(f1_m)

res_df = pd.DataFrame(results)

print("=== Results (sorted by COVID F1 desc) ===")
print(res_df.sort_values(["covid_f1", "covid_recall", "accuracy"], ascending=False).to_string(index=False))


# -------------------------
# STEP 7: Ploting curves
# -------------------------
def plot_metric_curve(metric_label: str, y_col: str, title: str, y_label: str):
    sub_e = res_df[res_df["metric"] == "euclidean"].sort_values("k")
    sub_m = res_df[res_df["metric"] == "manhattan"].sort_values("k")

    plt.figure()
    plt.plot(sub_e["k"], sub_e[y_col], marker="o", label="euclidean")
    plt.plot(sub_m["k"], sub_m[y_col], marker="o", label="manhattan")
    plt.xlabel("k")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_metric_curve(
    metric_label="both",
    y_col="accuracy",
    title="KNN Accuracy vs k (Euclidean vs Manhattan)",
    y_label="Accuracy"
)

plot_metric_curve(
    metric_label="both",
    y_col="covid_recall",
    title="KNN COVID Recall vs k (Euclidean vs Manhattan)",
    y_label=f"Recall (pos_label={pos_label})"
)

plot_metric_curve(
    metric_label="both",
    y_col="covid_f1",
    title="KNN COVID F1 vs k (Euclidean vs Manhattan)",
    y_label=f"F1-score (pos_label={pos_label})"
)


# -------------------------
# STEP 8: print best model details
# -------------------------
best = res_df.sort_values(["covid_f1", "covid_recall", "accuracy"], ascending=False).iloc[0]
best_k = int(best["k"])
best_metric = best["metric"]

print("\n=== Best setting by COVID F1 (then recall, then accuracy) ===")
print(best.to_string())

p_best = 2 if best_metric == "euclidean" else 1
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=p_best)
best_knn.fit(X_train_scaled, y_train)
best_pred = best_knn.predict(X_test_scaled)

print("\nConfusion matrix (best):")
print(confusion_matrix(y_test, best_pred))

print("\nClassification report (best):")
print(classification_report(y_test, best_pred, zero_division=0))
