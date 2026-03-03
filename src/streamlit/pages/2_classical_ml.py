from pathlib import Path
import sys
from typing import Optional, Tuple, List

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.metrics import classification_report

# =========================================================
# PAGE CONFIG
# =========================================================
st.title("Classical ML")

# =========================================================
# PATHS (robust, repo-relative)
# __file__ = .../src/streamlit/pages/2_classical_ml.py
# repo root = parents[2]
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent  # .../src/streamlit/pages
BASE_DIR = CURRENT_DIR.parents[2]  # .../nov25_bds_int_covid1 (repo root)

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed" / "ML" / "test_data"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Ensure repo root is importable (for "from models import ...")
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Optional helper imports (may contain hardcoded old paths -> fallback below)
KNN_HELPER_IMPORT_ERROR = None
XGB_HELPER_IMPORT_ERROR = None
knn_pickle_run = None
xgb_pickle_run = None

try:
    from models import knn_pickle_run as _knn_pickle_run  # type: ignore

    knn_pickle_run = _knn_pickle_run
except Exception as e:
    KNN_HELPER_IMPORT_ERROR = e

try:
    from models import xgb_pickle_run as _xgb_pickle_run  # type: ignore

    xgb_pickle_run = _xgb_pickle_run
except Exception as e:
    XGB_HELPER_IMPORT_ERROR = e


# =========================================================
# FILE CHECKS (fail gracefully for core page artifacts)
# =========================================================
def require_core_files():
    required = [
        DATA_DIR / "X_test.csv",
        DATA_DIR / "y_test.csv",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "logistic_none.pkl",
        MODELS_DIR / "logistic_oversample.pkl",
        MODELS_DIR / "logistic_smote.pkl",
        MODELS_DIR / "logistic_undersample.pkl",
        MODELS_DIR / "rf_none.pkl",
        MODELS_DIR / "rf_oversample.pkl",
        MODELS_DIR / "rf_smote.pkl",
        MODELS_DIR / "rf_undersample.pkl",
        MODELS_DIR / "svc_undersample.pkl",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        st.error("Missing required local files for the Classical ML page.")
        st.write("Please check that the following files exist:")
        for p in missing:
            st.write(f"- `{p}`")
        with st.expander("Resolved paths (debug)", expanded=True):
            st.write(f"CURRENT_DIR: `{CURRENT_DIR}`")
            st.write(f"BASE_DIR: `{BASE_DIR}`")
            st.write(f"MODELS_DIR: `{MODELS_DIR}`")
            st.write(f"DATA_DIR: `{DATA_DIR}`")
            st.write(f"FIGURES_DIR: `{FIGURES_DIR}`")
        st.stop()


require_core_files()


# =========================================================
# HELPERS
# =========================================================
def load_test_data_and_scaler():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test_df = pd.read_csv(DATA_DIR / "y_test.csv")
    y_test = y_test_df.iloc[:, 0]  # ensure 1D target series
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    X_test_scaled = scaler.transform(X_test)
    return X_test, y_test, X_test_scaled


def _find_covid_key(report_dict: dict) -> Optional[str]:
    """Robustly detect the positive/COVID class key in classification_report output."""
    preferred = ["COVID", "covid", "1", "1.0", "True", "positive", "Positive"]
    for k in preferred:
        if k in report_dict:
            return k

    # heuristic fallback: first non-aggregate class row
    aggregates = {"accuracy", "macro avg", "weighted avg", "samples avg"}
    class_keys = [k for k in report_dict.keys() if k not in aggregates]
    if len(class_keys) == 1:
        return class_keys[0]
    if len(class_keys) >= 2:
        # Prefer a key that looks like class 1 / covid
        for k in class_keys:
            s = str(k).lower()
            if "cov" in s or s in {"1", "1.0"}:
                return k
        # final fallback: second class key often positive class in binary reports
        return class_keys[-1]
    return None


def _extract_covid_metrics(report_dict: dict) -> dict:
    k = _find_covid_key(report_dict)
    if k is None:
        raise KeyError(
            "Could not find class row in classification_report output. "
            f"Available keys: {list(report_dict.keys())}"
        )
    return {
        "recall": float(report_dict[k]["recall"]),
        "precision": float(report_dict[k]["precision"]),
        "class_key": str(k),
    }


def _safe_crosstab(y_true, y_pred):
    return pd.crosstab(
        y_true,
        y_pred,
        rownames=["Real Class"],
        colnames=["Pred Class"],
    )


def _align_y_pred_to_y_true(y_true, y_pred):
    """
    Make y_pred label type compatible with y_true for metrics/crosstab.
    Common case here: y_true = ['COVID','Non-COVID'], y_pred = [0,1]
    """
    y_true_s = pd.Series(y_true).copy()
    y_pred_s = pd.Series(y_pred).copy()

    # If true labels are strings and preds are numeric -> map 0/1 to class names
    if y_true_s.dtype == object and pd.api.types.is_numeric_dtype(y_pred_s):
        int_to_str = {0: "Non-COVID", 1: "COVID"}
        mapped = y_pred_s.map(int_to_str)

        # fallback if unexpected values occur
        if mapped.isna().any():
            mapped = mapped.fillna(y_pred_s.astype(str))

        y_pred_s = mapped

    # If true labels are numeric and preds are strings -> map to 0/1
    elif pd.api.types.is_numeric_dtype(y_true_s) and y_pred_s.dtype == object:
        str_to_int = {
            "non-covid": 0,
            "normal": 0,
            "covid": 1,
            "covid-19": 1,
        }
        y_pred_s = y_pred_s.astype(str).str.strip().str.lower().map(str_to_int)

    return y_true_s, y_pred_s


def _extract_model_with_predict(obj):
    """Extract actual estimator from loaded pickle/bundle/list/dict."""
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in ["model", "classifier", "estimator", "xgb_model", "pipeline"]:
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if hasattr(item, "predict"):
                return item

    raise TypeError(
        f"Loaded object is not a model with .predict(). Got type: {type(obj)}"
    )


def _try_helper_knn_predict() -> Tuple[dict, pd.DataFrame]:
    if knn_pickle_run is None:
        raise RuntimeError(f"KNN helper import failed: {KNN_HELPER_IMPORT_ERROR}")
    return knn_pickle_run.predict()


def _predict_knn_local() -> Tuple[dict, pd.DataFrame]:
    """Fallback KNN prediction if helper module has hardcoded paths."""
    model_path = MODELS_DIR / "knn_model_ver_2.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    x_test_path = DATA_DIR / "X_test.csv"
    y_test_path = DATA_DIR / "y_test.csv"

    required = [model_path, scaler_path, x_test_path, y_test_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing KNN fallback files:\n" + "\n".join(missing))

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    # Align labels for metrics/crosstab (e.g., y_test strings vs y_pred 0/1)
    y_true_eval, y_pred_eval = _align_y_pred_to_y_true(y_test, y_pred)

    report = classification_report(y_true_eval, y_pred_eval, output_dict=True)
    ct = _safe_crosstab(y_true_eval, y_pred_eval)
    return report, ct


def safe_knn_predict() -> Tuple[dict, pd.DataFrame]:
    try:
        return _try_helper_knn_predict()
    except Exception:
        return _predict_knn_local()


def _xgb_model_candidates() -> List[Path]:
    patterns = ["xgb*.pkl", "xgboost*.pkl", "*xgb*.pkl"]
    cands: List[Path] = []

    for pat in patterns:
        cands.extend(MODELS_DIR.glob(pat))

    # dedupe + filter obvious non-model artifacts
    excluded_keywords = [
        "feature",
        "features",
        "columns",
        "scaler",
        "metrics",
        "proba",
        "report",
    ]

    cleaned: List[Path] = []
    for p in sorted(set(cands)):
        name_lower = p.name.lower()

        if p.suffix != ".pkl":
            continue
        if name_lower == "xgb_pickle_run.py":
            continue
        if any(k in name_lower for k in excluded_keywords):
            continue

        cleaned.append(p)

    # rank likely model names first
    def _score(path: Path) -> int:
        n = path.name.lower()
        if n == "xgboost_model.pkl":
            return 0
        if n == "xgb_model.pkl":
            return 1
        if "model" in n:
            return 2
        return 10

    cleaned = sorted(cleaned, key=_score)
    return cleaned


def _predict_xgb_local(threshold: float = 0.5) -> Tuple[dict, pd.DataFrame]:
    """
    Fallback XGBoost prediction:
    - locates an XGB model pickle in /models
    - extracts estimator if pickle contains bundle/list/dict
    - uses predict_proba if available and applies threshold
    - aligns labels (y_true strings vs y_pred ints) for metrics/crosstab
    """
    xgb_candidates = _xgb_model_candidates()
    if not xgb_candidates:
        raise FileNotFoundError(
            "No XGBoost pickle found in models/. Expected something like xgb*.pkl"
        )

    model_path = xgb_candidates[0]  # best-ranked candidate
    scaler_path = MODELS_DIR / "scaler.pkl"
    x_test_path = DATA_DIR / "X_test.csv"
    y_test_path = DATA_DIR / "y_test.csv"

    required = [model_path, scaler_path, x_test_path, y_test_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing XGB fallback files:\n" + "\n".join(missing))

    # Load model (may be actual model OR bundle/list/dict)
    loaded_obj = joblib.load(model_path)
    model = _extract_model_with_predict(loaded_obj)

    scaler = joblib.load(scaler_path)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    X_test_scaled = scaler.transform(X_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_scaled)

        # binary classification -> positive class usually column 1
        if (
            getattr(proba, "shape", None) is not None
            and len(proba.shape) > 1
            and proba.shape[1] >= 2
        ):
            p_pos = proba[:, 1]
        else:
            p_pos = pd.Series(proba).astype(float).values

        y_pred = (p_pos >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)

    # Align labels for metrics/crosstab (e.g. y_test strings vs y_pred 0/1)
    y_true_eval, y_pred_eval = _align_y_pred_to_y_true(y_test, y_pred)

    report = classification_report(y_true_eval, y_pred_eval, output_dict=True)
    ct = _safe_crosstab(y_true_eval, y_pred_eval)
    return report, ct


def _try_helper_xgb_predict(threshold: float = 0.5) -> Tuple[dict, pd.DataFrame]:
    if xgb_pickle_run is None:
        raise RuntimeError(f"XGB helper import failed: {XGB_HELPER_IMPORT_ERROR}")
    return xgb_pickle_run.predict(threshold=threshold)


def safe_xgb_predict(threshold: float = 0.5) -> Tuple[dict, pd.DataFrame]:
    try:
        return _try_helper_xgb_predict(threshold=threshold)
    except Exception:
        return _predict_xgb_local(threshold=threshold)


def display_cr_cm(report_dict, ct):
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)
    st.write("")

    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    # ✅ CM bewusst kompakt halten
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    sns.heatmap(
        ct_norm,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
    plt.close(fig)


def get_report_ct(classifier, clf, X_test_scaled, y_test):
    if any(x in classifier for x in ["Logistic regression", "Random forest", "SVC"]):
        y_pred = clf.predict(X_test_scaled)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        ct = _safe_crosstab(y_test, y_pred)

    elif classifier == "KNN":
        report_dict, ct = safe_knn_predict()

    elif classifier == "XGBoost_0.5":
        report_dict, ct = safe_xgb_predict(threshold=0.5)

    elif classifier == "XGBoost_0.3":
        report_dict, ct = safe_xgb_predict(threshold=0.3)

    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return report_dict, ct


@st.cache_data(show_spinner=False)
def get_all_baseline_metrics():
    """Calculate metrics for all models once and cache them."""
    _, y_test, X_test_scaled = load_test_data_and_scaler()

    results = []

    # Logistic & RF variants
    configs = {"Logistic regression": "logistic", "Random forest": "rf"}
    methods = {
        "None": "none.pkl",
        "Oversample": "oversample.pkl",
        "Smote": "smote.pkl",
        "Undersample": "undersample.pkl",
    }

    for label, prefix in configs.items():
        for res_name, file in methods.items():
            clf = joblib.load(MODELS_DIR / f"{prefix}_{file}")
            y_pred = clf.predict(X_test_scaled)
            rep = classification_report(y_test, y_pred, output_dict=True)
            m = _extract_covid_metrics(rep)

            results.append(
                {
                    "model": f"{label} ({res_name})",
                    "type": label,
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "class_key": m["class_key"],
                }
            )

    # SVC
    svc_clf = joblib.load(MODELS_DIR / "svc_undersample.pkl")
    svc_y_pred = svc_clf.predict(X_test_scaled)
    svc_rep = classification_report(y_test, svc_y_pred, output_dict=True)
    svc_m = _extract_covid_metrics(svc_rep)
    results.append(
        {
            "model": "SVC",
            "type": "SVC",
            "recall": svc_m["recall"],
            "precision": svc_m["precision"],
            "class_key": svc_m["class_key"],
        }
    )

    # KNN (helper or local fallback)
    try:
        knn_rep, _ = safe_knn_predict()
        knn_m = _extract_covid_metrics(knn_rep)
        results.append(
            {
                "model": "KNN",
                "type": "KNN",
                "recall": knn_m["recall"],
                "precision": knn_m["precision"],
                "class_key": knn_m["class_key"],
            }
        )
    except Exception as e:
        # Do not crash the page; show later in debug
        results.append(
            {
                "model": "KNN (unavailable)",
                "type": "KNN",
                "recall": None,
                "precision": None,
                "class_key": f"ERROR: {e}",
            }
        )

    # XGBoost variants (helper or local fallback)
    for t in [0.5, 0.3]:
        try:
            xgb_rep, _ = safe_xgb_predict(threshold=t)
            xgb_m = _extract_covid_metrics(xgb_rep)
            results.append(
                {
                    "model": f"XGBoost_{t}",
                    "type": "XGBoost",
                    "recall": xgb_m["recall"],
                    "precision": xgb_m["precision"],
                    "class_key": xgb_m["class_key"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "model": f"XGBoost_{t} (unavailable)",
                    "type": "XGBoost",
                    "recall": None,
                    "precision": None,
                    "class_key": f"ERROR: {e}",
                }
            )

    df = pd.DataFrame(results)

    # chart-safe subset with valid metrics only
    if "recall" in df.columns:
        df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    if "precision" in df.columns:
        df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    return df


# =========================================================
# CACHE GLOBAL METRICS
# =========================================================
all_metrics_df = get_all_baseline_metrics()

# =========================================================
# UI
# =========================================================
st.subheader("Covid-19 Diagnosis Using Chest X-Rays")
st.sidebar.title("Table of contents")
pages = ["Baseline Models", "Summary"]
page = st.sidebar.radio("Go to", pages)

# Optional debug info
with st.sidebar.expander("Debug paths", expanded=False):
    st.write(f"BASE_DIR: `{BASE_DIR}`")
    st.write(f"MODELS_DIR: `{MODELS_DIR}`")
    st.write(f"DATA_DIR: `{DATA_DIR}`")
    st.write(f"FIGURES_DIR: `{FIGURES_DIR}`")
    st.write(f"KNN helper import error: `{KNN_HELPER_IMPORT_ERROR}`")
    st.write(f"XGB helper import error: `{XGB_HELPER_IMPORT_ERROR}`")
    st.write("XGB model candidates found:")
    for p in _xgb_model_candidates():
        st.write(f"- `{p}`")

# Any unavailable-model rows -> visible hint
unavailable_rows = all_metrics_df[
    all_metrics_df["model"].astype(str).str.contains(r"\(unavailable\)", regex=True)
]
if not unavailable_rows.empty:
    with st.expander("Model availability warnings", expanded=False):
        for _, row in unavailable_rows.iterrows():
            st.warning(f"{row['model']}: {row.get('class_key', 'Unknown error')}")


if page == "Baseline Models":
    X_test, y_test, X_test_scaled = load_test_data_and_scaler()

    def prediction(classifier):
        if classifier in ["Logistic regression", "Random forest"]:
            resampling_map = {
                "None": "none.pkl",
                "Oversample": "oversample.pkl",
                "Smote": "smote.pkl",
                "Undersample": "undersample.pkl",
            }
            prefix = "logistic" if classifier == "Logistic regression" else "rf"

            choice = st.radio(
                f"Which {classifier} re-sampling method?",
                list(resampling_map.keys()),
            )
            model_name = f"{prefix}_{resampling_map[choice]}"

            report, ct = get_report_ct(
                classifier,
                joblib.load(MODELS_DIR / model_name),
                X_test_scaled,
                y_test,
            )
            display_cr_cm(report, ct)

            st.write("")
            st.write("")

            current_df = all_metrics_df[
                (all_metrics_df["type"] == classifier)
                & all_metrics_df["recall"].notna()
                & all_metrics_df["precision"].notna()
            ]
            if current_df.empty:
                st.info("No metrics available for this model group.")
                return

            chart = (
                alt.Chart(current_df.round(2))
                .transform_calculate(
                    jitter_x="datum.recall + (random() - 0.5) * 0.02",
                    jitter_y="datum.precision + (random() - 0.5) * 0.02",
                )
                .mark_circle(size=100, opacity=0.7)
                .encode(
                    x=alt.X(
                        "jitter_x:Q",
                        scale=alt.Scale(domain=[-0.05, 1.05]),
                        title="Recall (COVID)",
                    ),
                    y=alt.Y(
                        "jitter_y:Q",
                        scale=alt.Scale(domain=[-0.05, 1.05]),
                        title="Precision (COVID)",
                    ),
                    color=alt.Color(
                        "model",
                        legend=alt.Legend(orient="bottom", columns=2, symbolLimit=0),
                    ),
                    tooltip=["model", "recall", "precision"],
                )
                .properties(height=450)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        elif classifier == "KNN":
            try:
                report, ct = get_report_ct(classifier, None, X_test_scaled, y_test)
            except Exception as e:
                st.error(f"KNN prediction failed: {e}")
                st.info(
                    "Likely cause: hardcoded path in models/knn_pickle_run.py. "
                    "This page tries a fallback, but if the model filename differs, update the fallback path."
                )
                return

            display_cr_cm(report, ct)

            # Optional figure files (do not crash if missing)
            for fname in ["knn_accuracy.png", "knn_recall.png", "knn_f1.png"]:
                fpath = FIGURES_DIR / fname
                if fpath.exists():
                    st.image(str(fpath))
                else:
                    st.warning(f"Missing figure: {fpath}")

        elif classifier == "SVC":
            st.write("##### C = 10, gamma = 0.1, kernel = rbf")
            report, ct = get_report_ct(
                classifier,
                joblib.load(MODELS_DIR / "svc_undersample.pkl"),
                X_test_scaled,
                y_test,
            )
            display_cr_cm(report, ct)

        elif classifier == "XGBoost":
            threshold_map = {"0.5": "XGBoost_0.5", "0.3": "XGBoost_0.3"}
            choice = st.radio("Which threshold?", list(threshold_map.keys()))

            try:
                report, ct = get_report_ct(
                    threshold_map[choice],
                    None,
                    X_test_scaled,
                    y_test,
                )
            except Exception as e:
                st.error(f"XGBoost prediction failed: {e}")
                st.info(
                    "Likely cause: hardcoded path in models/xgb_pickle_run.py or unexpected XGB model filename. "
                    "Check the sidebar debug panel for detected xgb*.pkl files."
                )
                return

            display_cr_cm(report, ct)
            st.write("")
            st.write("")

            xgb_df = all_metrics_df[
                (all_metrics_df["type"] == "XGBoost")
                & all_metrics_df["recall"].notna()
                & all_metrics_df["precision"].notna()
                & ~all_metrics_df["model"]
                .astype(str)
                .str.contains(r"\(unavailable\)", regex=True)
            ]
            if xgb_df.empty:
                st.info("No XGBoost metrics available.")
                return

            chart = (
                alt.Chart(xgb_df.round(2))
                .mark_circle(size=100)
                .encode(
                    x=alt.X(
                        "recall",
                        scale=alt.Scale(domain=[-0.05, 1.05]),
                        title="Recall (COVID)",
                    ),
                    y=alt.Y(
                        "precision",
                        scale=alt.Scale(domain=[-0.05, 1.05]),
                        title="Precision (COVID)",
                    ),
                    color=alt.Color(
                        "model", legend=alt.Legend(orient="bottom", columns=2)
                    ),
                    tooltip=["model", "recall", "precision"],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

    choice_list = ["Logistic regression", "KNN", "Random forest", "SVC", "XGBoost"]
    option = st.selectbox("Choice of the model", choice_list)
    prediction(option)

if page == "Summary":
    st.write("### Global Model Comparison")

    chart_df = all_metrics_df[
        all_metrics_df["recall"].notna() & all_metrics_df["precision"].notna()
    ].copy()

    if chart_df.empty:
        st.warning("No model metrics available to display.")
    else:
        chart = (
            alt.Chart(chart_df.round(2))
            .transform_calculate(
                jitter_x="datum.recall + (random() - 0.5) * 0.02",
                jitter_y="datum.precision + (random() - 0.5) * 0.02",
            )
            .mark_circle(size=100, opacity=0.7)
            .encode(
                x=alt.X(
                    "jitter_x:Q",
                    scale=alt.Scale(domain=[-0.05, 1.05]),
                    title="Recall (COVID)",
                ),
                y=alt.Y(
                    "jitter_y:Q",
                    scale=alt.Scale(domain=[-0.05, 1.05]),
                    title="Precision (COVID)",
                ),
                color=alt.Color(
                    "model",
                    legend=alt.Legend(orient="bottom", columns=2, symbolLimit=0),
                ),
                tooltip=["model", "recall", "precision"],
            )
            .properties(height=450)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Metrics table (debug)")
    st.dataframe(all_metrics_df, use_container_width=True)
