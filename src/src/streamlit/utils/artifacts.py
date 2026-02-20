from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_metrics_summary(art_dir: Path) -> pd.DataFrame:
    p = art_dir / "metrics_summary.csv"
    if p.exists():
        return pd.read_csv(p)

    # fallback demo values
    return pd.DataFrame(
        [
            {
                "model": "cnn",
                "accuracy": 0.96,
                "precision": 0.87,
                "recall": 0.91,
                "f1": 0.89,
                "macro_f1": 0.93,
                "auc_pr": 0.92,
                "auc_roc": 0.97,
            },
            {
                "model": "resnet",
                "accuracy": 0.97,
                "precision": 0.89,
                "recall": 0.92,
                "f1": 0.90,
                "macro_f1": 0.94,
                "auc_pr": 0.94,
                "auc_roc": 0.98,
            },
        ]
    )


def load_confusion_matrices(art_dir: Path) -> dict:
    p = art_dir / "confusion_matrices.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # format: [[TN, FP], [FN, TP]]
    return {
        "cnn": [[180, 12], [9, 95]],
        "resnet": [[184, 8], [8, 96]],
    }


def load_pr_curves(art_dir: Path) -> dict:
    p = art_dir / "pr_curves.npz"
    if p.exists():
        arr = np.load(p)
        return {
            "cnn_precision": arr["cnn_precision"],
            "cnn_recall": arr["cnn_recall"],
            "resnet_precision": arr["resnet_precision"],
            "resnet_recall": arr["resnet_recall"],
        }

    # fallback synthetic curves
    r = np.linspace(0, 1, 200)
    return {
        "cnn_recall": r,
        "cnn_precision": np.clip(1 - (r**1.6), 0, 1),
        "resnet_recall": r,
        "resnet_precision": np.clip(1 - (r**2.0), 0, 1),
    }


def load_roc_curves(art_dir: Path) -> dict:
    p = art_dir / "roc_curves.npz"
    if p.exists():
        arr = np.load(p)
        return {
            "cnn_fpr": arr["cnn_fpr"],
            "cnn_tpr": arr["cnn_tpr"],
            "resnet_fpr": arr["resnet_fpr"],
            "resnet_tpr": arr["resnet_tpr"],
        }

    # fallback synthetic curves
    x = np.linspace(0, 1, 200)
    return {
        "cnn_fpr": x,
        "cnn_tpr": np.clip(x**0.45, 0, 1),
        "resnet_fpr": x,
        "resnet_tpr": np.clip(x**0.35, 0, 1),
    }


def load_history(art_dir: Path, model_name: str) -> pd.DataFrame:
    p = art_dir / f"history_{model_name}.csv"
    if p.exists():
        df = pd.read_csv(p)
        # expected columns: epoch,train_loss,val_loss,val_f1
        return df

    # fallback synthetic history
    epochs = np.arange(1, 31)
    if model_name == "cnn":
        train_loss = np.exp(-epochs / 12) + 0.08
        val_loss = np.exp(-epochs / 10) + 0.12
        val_f1 = 0.55 + 0.40 * (1 - np.exp(-epochs / 8))
    else:
        train_loss = np.exp(-epochs / 13) + 0.07
        val_loss = np.exp(-epochs / 11) + 0.10
        val_f1 = 0.58 + 0.38 * (1 - np.exp(-epochs / 7))

    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": np.clip(val_f1, 0, 1),
        }
    )
