# src/explain.py
"""
SHAP-based model interpretability.

BUGS FIXED FROM THE PREVIOUS VERSION:
1. `isinstance(model, (RandomForestClassifier, XGBClassifier))` referenced both
   classes without importing them - guaranteed NameError on the very first call.
2. Since train.py now returns a sklearn `Pipeline` (StandardScaler + clf) rather
   than a bare estimator, `shap.TreeExplainer(model)` / `LinearExplainer(model, ...)`
   need the underlying `clf` step, not the whole Pipeline - and for the linear
   model, the *scaled* data, not raw X_test. Both are handled below.
3. `explainer.expected_value` can be a scalar, a 2-element array, or a list
   depending on the SHAP/XGBoost version - indexing it blindly
   (`explainer.expected_value` alone, or always `[1]`) breaks depending on
   version. This version normalizes it to a single scalar for the positive class.
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from utils import setup_logging, MODELS_DIR, FIGURES_DIR, ensure_all_project_dirs

logger = logging.getLogger(__name__)


def _unwrap_estimator(model):
    """Pulls the actual classifier + any preceding scaler out of a Pipeline."""
    if isinstance(model, Pipeline):
        clf = model.named_steps.get('clf', model.steps[-1][1])
        scaler = model.named_steps.get('scaler')
        return clf, scaler
    return model, None


def _normalize_expected_value(expected_value):
    """Collapses SHAP's expected_value (scalar / list / array) to one float for the positive class."""
    if isinstance(expected_value, (list, np.ndarray)):
        arr = np.asarray(expected_value).flatten()
        return float(arr[-1])  # positive class is the last entry when there are 2
    return float(expected_value)


def build_explainer(model, X_background: pd.DataFrame):
    """Returns (explainer, transformed_background) appropriate for the model type."""
    clf, scaler = _unwrap_estimator(model)
    X_for_explainer = scaler.transform(X_background) if scaler is not None else X_background

    if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(clf)
    elif isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_for_explainer)
    else:
        # Fallback: model-agnostic explainer, works for anything with predict_proba
        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        explainer = shap.KernelExplainer(predict_fn, shap.sample(X_background, min(100, len(X_background))))

    return explainer, X_for_explainer


def generate_shap_values(model, X_test: pd.DataFrame):
    """Generates SHAP values for the positive (adverse outcome) class."""
    explainer, X_transformed = build_explainer(model, X_test)
    raw_shap_values = explainer.shap_values(X_transformed)

    # Handle the "list per class" format some SHAP/model combos still return.
    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[-1]
    else:
        shap_values = raw_shap_values

    expected_value = _normalize_expected_value(explainer.expected_value)
    return explainer, shap_values, expected_value, X_transformed


def plot_summary(shap_values, X_display: pd.DataFrame, filename: str = 'shap_summary.png'):
    """Plots the global SHAP summary plot."""
    ensure_all_project_dirs()
    plt.figure()
    shap.summary_plot(shap_values, X_display, show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    logger.info(f"Saved SHAP summary plot to {FIGURES_DIR / filename}")


def get_patient_explanation(explainer, shap_values, expected_value, patient_idx: int,
                             X_display: pd.DataFrame, feature_names=None):
    """Generates a waterfall plot for an individual patient."""
    ensure_all_project_dirs()
    feature_names = feature_names or list(X_display.columns)
    row_values = X_display.iloc[patient_idx].values if hasattr(X_display, 'iloc') else X_display[patient_idx]

    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=expected_value,
        data=row_values,
        feature_names=feature_names,
    )

    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"Patient {patient_idx} Risk Contribution")
    plt.tight_layout()
    out_path = FIGURES_DIR / f'shap_patient_{patient_idx}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved patient explanation to {out_path}")
    return explanation


if __name__ == "__main__":
    setup_logging()
    model = joblib.load(MODELS_DIR / 'xgb_risk_model.pkl')
    X_test, y_test = joblib.load(MODELS_DIR / 'test_set.pkl')

    # SHAP on a sample for speed (full test sets can be slow, especially for KernelExplainer fallback)
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer, shap_values, expected_value, X_transformed = generate_shap_values(model, sample)

    plot_summary(shap_values, sample)
    get_patient_explanation(explainer, shap_values, expected_value, 0, sample)
