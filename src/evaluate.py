"""
Evaluate one or multiple saved fraud detection models on a CSV.

Adds evaluate_all_models() that will look for these bundles under models/:
- rf_bundle.joblib (preferred for RandomForest)
- RandomForest_bundle.joblib
- LogisticRegression_bundle.joblib
- XGBoost_bundle.joblib

Each bundle is expected to be a dict with keys: {'model', 'scaler', 'features'}
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data_processing import feature_engineer


MODELS_DIR = os.path.join("models")


def _load_bundle(path: str):
    """Load a saved model bundle if present, else return None."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        warnings.warn(f"Failed to load bundle {path}: {e}")
        return None


def _prepare_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Return DataFrame with exactly the columns in features, in order.

    - Applies feature_engineer to ensure engineered columns exist.
    - Keeps only numeric columns per the saved feature list.
    - For any missing engineered feature, fill with zeros to allow evaluation.
    """
    df_fe = feature_engineer(df.copy())
    # Ensure all expected columns exist
    for col in features:
        if col not in df_fe.columns:
            # create missing column as zeros (safe fallback for engineered features)
            df_fe[col] = 0
    X = df_fe[features]
    # keep numeric only (just in case)
    X = X.select_dtypes(include=[np.number])
    # after selection, if order changed due to dtype filtering, reindex
    X = X.reindex(columns=features)
    return X


def eval_bundle_on_csv(bundle: Dict, df_path: str, label_col: str = "Class") -> Tuple[Dict, Dict]:
    """Evaluate a single loaded bundle on a CSV.

    Returns (metrics_dict, info_dict)
    """
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    features: List[str] = bundle.get("features", [])

    df = pd.read_csv(df_path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {df_path}")

    X = _prepare_features(df, features)
    y = df[label_col].astype(int)

    # Align lengths in case any rows were dropped (shouldn't be, but safe)
    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]

    # Scale
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    # Predict
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xs)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(Xs)
        s_min, s_max = s.min(), s.max()
        y_proba = (s - s_min) / (s_max - s_min + 1e-9)
    else:
        # Fallback to predictions as probabilities (not ideal)
        preds = model.predict(Xs)
        y_proba = preds.astype(float)

    y_pred = (y_proba >= 0.5).astype(int) if not hasattr(model, "predict") else model.predict(Xs)

    report = classification_report(y, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    roc = roc_auc_score(y, y_proba)

    metrics = {
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
    info = {
        "n_rows": int(len(df)),
        "n_used": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": features,
    }
    return metrics, info


def evaluate_all_models(df_path: str = "data/creditcard1.csv", label_col: str = "Class") -> None:
    """Evaluate multiple saved models if present and print a summary."""
    candidates = [
        os.path.join(MODELS_DIR, "rf_bundle.joblib"),
        os.path.join(MODELS_DIR, "RandomForest_bundle.joblib"),
        os.path.join(MODELS_DIR, "LogisticRegression_bundle.joblib"),
        os.path.join(MODELS_DIR, "XGBoost_bundle.joblib"),
    ]

    loaded = []
    for path in candidates:
        bundle = _load_bundle(path)
        if bundle is not None:
            loaded.append((os.path.basename(path), bundle))

    if not loaded:
        print("No model bundles found under 'models/'. Train a model first.")
        return

    summary_rows = []
    for name, bundle in loaded:
        print(f"\n===== Evaluating {name} =====")
        try:
            metrics, info = eval_bundle_on_csv(bundle, df_path, label_col=label_col)
            print(metrics["report"])  # classification report
            print("Confusion Matrix:", np.array(metrics["confusion_matrix"]))
            print("ROC AUC:", metrics["roc_auc"])
            summary_rows.append({
                "Model": name,
                "ROC_AUC": metrics["roc_auc"],
                "Rows": info["n_used"],
                "Features": info["n_features"],
            })
        except Exception as e:
            warnings.warn(f"Evaluation failed for {name}: {e}")

    if summary_rows:
        print("\n===== Summary =====")
        summary_df = pd.DataFrame(summary_rows).sort_values("ROC_AUC", ascending=False)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # Run: python -m src.evaluate
    evaluate_all_models(df_path="data/creditcard1.csv", label_col="Class")

