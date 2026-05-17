"""
logic.py — Streamlit logic layer
Loads the pre-trained model bundle (heart_model_bundle.pkl)
produced by heart_disease_training.ipynb (Colab).

No training happens here — Streamlit only loads and predicts.
"""

import joblib
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. LOAD BUNDLE
# ─────────────────────────────────────────────

def load_bundle(path: str = "heart_model_bundle.pkl") -> dict:
    """
    Load the pre-trained model bundle saved from Colab.
    Returns the full bundle dict.
    """
    bundle = joblib.load(path)
    return bundle


# ─────────────────────────────────────────────
# 2. DATASET SUMMARY (for EDA display)
# ─────────────────────────────────────────────

def get_dataset_summary(bundle: dict) -> dict:
    """Extract dataset info stored inside the bundle."""
    return {
        "train_samples" : bundle["X_train_shape"][0],
        "test_samples"  : bundle["X_test_shape"][0],
        "n_features"    : bundle["X_train_shape"][1],
        "class_dist"    : bundle["class_dist"],
        "feature_cols"  : bundle["feature_cols"],
        "numerical"     : bundle["numerical"],
        "categorical"   : bundle["categorical"],
    }


# ─────────────────────────────────────────────
# 3. METRICS HELPERS
# ─────────────────────────────────────────────

def get_comparison_df(bundle: dict) -> pd.DataFrame:
    """Return the model comparison DataFrame from the bundle."""
    display_cols = [
        "Model", "Accuracy Train", "Accuracy Test",
        "Precision", "Recall", "F1-Score",
        "ROC-AUC", "CV F1 Mean", "CV F1 Std", "Fit Status",
    ]
    df = bundle["comparison_df"]
    return df[[c for c in display_cols if c in df.columns]]


def get_best_metrics(bundle: dict) -> dict:
    """Return metrics dict for the best model."""
    name = bundle["model_name"]
    return bundle["metrics"][name]


def get_all_metrics(bundle: dict) -> dict:
    """Return metrics dict for all models."""
    return bundle["metrics"]


def get_confusion_matrices(bundle: dict) -> dict:
    """Return {model_name: confusion_matrix} dict."""
    return bundle["confusion_matrices"]


def get_roc_data(bundle: dict) -> dict:
    """Return roc_data and auc scores."""
    return bundle["roc_data"], bundle["roc_auc"]


def get_feature_importances(bundle: dict) -> pd.Series:
    """Return feature importances from Random Forest, sorted descending."""
    fi = bundle["feature_importances"]
    return pd.Series(fi).sort_values(ascending=False)


# ─────────────────────────────────────────────
# 4. SINGLE-PATIENT PREDICTION
# ─────────────────────────────────────────────

def predict_single(bundle: dict, input_dict: dict):
    """
    Predict heart disease for one patient.

    Parameters
    ----------
    bundle     : loaded pkl bundle
    input_dict : {feature_name: raw_value} with original column names

    Returns
    -------
    prediction : int  (0 = No Disease, 1 = Heart Disease)
    probability: float (confidence for predicted class)
    """
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    encoder_map  = bundle["encoder_map"]
    numerical    = bundle["numerical"]
    categorical  = bundle["categorical"]

    row = pd.DataFrame([input_dict])

    for col in categorical:
        le = encoder_map[col]
        row[col] = le.transform(row[col])

    row[numerical] = scaler.transform(row[numerical])

    prediction = int(model.predict(row)[0])
    try:
        prob = float(model.predict_proba(row)[0][prediction])
    except AttributeError:
        prob = None

    return prediction, prob
