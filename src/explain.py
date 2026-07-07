# src/explain.py
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

def generate_shap_explanations(model, X_test: pd.DataFrame, feature_names: list):
    """Generates SHAP values for model interpretability."""
    # For tree-based models, use TreeExplainer
    if hasattr(model, 'get_booster') or isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_test)
        
    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-class/multi-output SHAP formats if necessary
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Focus on the positive class (adverse outcome)
        
    return explainer, shap_values

def plot_summary(shap_values, X_test: pd.DataFrame):
    """Plots the global SHAP summary plot."""
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig('figures/shap_summary.png')
    plt.close()

def get_patient_explanation(explainer, shap_values, patient_idx: int, X_test: pd.DataFrame):
    """Generates a waterfall plot for an individual patient."""
    # Create an Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[patient_idx].values,
        feature_names=X_test.columns.tolist()
    )
    
    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"Patient {patient_idx} Risk Contribution")
    plt.tight_layout()
    plt.savefig(f'figures/shap_patient_{patient_idx}.png')
    plt.close()
    return explanation