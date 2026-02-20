
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 42
POS_LABEL = 1  # COVID = 1

# -------------------------
# STEP 1: Load data
# -------------------------
DATA_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\data\processed\ML\features_reduced_15.csv")
df = pd.read_csv(DATA_PATH)

print("Loaded:", df.shape)
print(df.head())

# -------------------------
# STEP 2: Target 
# -------------------------
y_raw = df["target"]

if y_raw.dtype == "object":
    y_clean = y_raw.astype(str).str.strip()

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

    if y.isna().any():
        unknown = sorted(y_clean[y.isna()].unique().tolist())
        print("\nERROR: Found target labels not covered by mapping:", unknown)
        print("Please add them to the mapping dictionary.")
        raise ValueError("Unmapped target labels present.")

    y = y.astype(int)
    print("\nTarget mapped successfully. Unique values:", sorted(y.unique().tolist()))

else:
    y = y_raw.astype(int)
    print("\nTarget already numeric. Unique values:", sorted(pd.Series(y).unique().tolist()))

# -------------------------
# STEP 3: Remove non-features and keep numeric only
# -------------------------
drop_cols = ["target", "image_label"]
drop_cols += ["image_path", "filename", "id", "image_name"]  # safe removals if they exist
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
X = X.select_dtypes(include=[np.number])

print("\nX shape:", X.shape)
print("Number of features:", X.shape[1])

if X.shape[1] == 0:
    raise ValueError("No numeric features found after filtering. Check your CSV columns.")

# -------------------------
# STEP 4: Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain size:", X_train.shape[0], " Test size:", X_test.shape[0])

# -------------------------
# STEP 5: Handle imbalance (binary)
# -------------------------
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

print(f"\nBinary problem. neg={neg}, pos={pos}, scale_pos_weight={scale_pos_weight:.3f}")

# -------------------------
# STEP 6: Build XGBoost model
# -------------------------
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1,
    gamma=0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss"
)

# -------------------------
# STEP 7: Train
# -------------------------
model.fit(X_train, y_train)
print("\nModel trained.")

# -------------------------
# STEP 8: Predict & Evaluate
# -------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Non-COVID", "COVID"],
        zero_division=0
    )
)

# -------------------------
# STEP 9: Extra metrics (binary)
# -------------------------
y_proba = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_proba)
pr = average_precision_score(y_test, y_proba)

print(f"\nROC-AUC: {roc:.4f}")
print(f"PR-AUC : {pr:.4f}")

# Optional: threshold tuning (improve recall)
threshold = 0.30
y_pred_thr = (y_proba >= threshold).astype(int)
print(f"\n--- Threshold evaluation (threshold={threshold}) ---")
print(confusion_matrix(y_test, y_pred_thr))
print(
    classification_report(
        y_test, y_pred_thr,
        labels=[0, 1],
        target_names=["Non-COVID", "COVID"],
        zero_division=0
    )
)

# -------------------------
# STEP 10: Save XGBoost model 
# -------------------------
MODEL_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\models\xgboost_model.pkl")
FEATURES_PATH = Path(r"C:\Projects\GitHub\nov25_bds_int_covid1\models\xgb_feature_columns.pkl")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(list(X.columns), FEATURES_PATH)

print(f"\nXGBoost model saved successfully to: {MODEL_PATH}")
print(f"Feature columns saved to: {FEATURES_PATH}")
