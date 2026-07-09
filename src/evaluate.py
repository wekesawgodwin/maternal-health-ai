# src/evaluate.py
"""
CRITICAL FIX: the previous version called `xgb_model.predict(X_test)`, which
applies sklearn's hardcoded 0.5 decision threshold. But train.py's whole
GridSearch + threshold-tuning step exists specifically because 0.5 catches
only ~55% of adverse outcomes on this imbalanced target - the deployed
threshold is ~0.30, saved in model_metadata.pkl. Evaluating at 0.5 would
silently report the WRONG recall for the model that's actually deployed.
This version loads model_metadata.pkl and applies its threshold to
predict_proba() instead of trusting .predict().
"""
import joblib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              average_precision_score, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import logging

from utils import setup_logging, load_model_metadata, MODELS_DIR, FIGURES_DIR, ensure_all_project_dirs

logger = logging.getLogger(__name__)


def evaluate_and_save_artifacts():
    ensure_all_project_dirs()
    logger.info("Loading model, test data, and tuned threshold...")

    model = joblib.load(MODELS_DIR / 'xgb_risk_model.pkl')
    X_test, y_test = joblib.load(MODELS_DIR / 'test_set.pkl')
    metadata = load_model_metadata(MODELS_DIR)
    threshold = metadata['threshold']
    model_name = metadata.get('model_name', 'unknown')

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)   # tuned threshold, NOT model.predict()'s 0.5

    # ==========================================
    # 1. METRICS (JSON)
    # ==========================================
    report = classification_report(y_test, y_pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model_name": model_name,
        "decision_threshold": round(float(threshold), 4),
        "recall": round(report['1']['recall'], 4),
        "precision": round(report['1']['precision'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "min_recall_target": metadata.get("min_recall_target"),
        "meets_recall_target": bool(report['1']['recall'] >= (metadata.get("min_recall_target") or 0)),
    }

    with open(MODELS_DIR / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {MODELS_DIR / 'evaluation_metrics.json'}")
    logger.info(f"Recall={metrics['recall']:.2%}  Precision={metrics['precision']:.2%}  "
                f"(threshold={threshold:.3f}, model={model_name})")

    # ==========================================
    # 2. PLOTS (PNG)
    # ==========================================
    logger.info("Generating evaluation plots...")

    # A. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Adverse'], yticklabels=['Normal', 'Adverse'])
    plt.title(f'Confusion Matrix (threshold={threshold:.3f})')
    plt.ylabel('Actual Outcome')
    plt.xlabel('Predicted Outcome')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eval_cm.png', dpi=150)
    plt.close()

    # B. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eval_roc.png', dpi=150)
    plt.close()

    # C. Precision-Recall Curve, with the DEPLOYED operating point marked
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prec_curve, color='green', lw=2, label=f'PR curve (AP = {metrics["pr_auc"]:.2f})')
    plt.scatter([metrics['recall']], [metrics['precision']], color='red', zorder=5,
                label=f'Deployed threshold ({threshold:.2f})')
    if metrics.get('min_recall_target'):
        plt.axvline(metrics['min_recall_target'], color='grey', linestyle='--', linewidth=1,
                    label=f"{metrics['min_recall_target']:.0%} recall target")
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eval_pr.png', dpi=150)
    plt.close()

    # D. Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='red', label='Model')
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.title('Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eval_calib.png', dpi=150)
    plt.close()

    # E. Model comparison, if train.py's model_comparison.json is present
    comp_path = MODELS_DIR / 'model_comparison.json'
    if comp_path.exists():
        with open(comp_path) as f:
            comparison = json.load(f)
        comp_df = pd.DataFrame({
            name: {'Recall': v['recall'], 'Precision': v['precision']} for name, v in comparison.items()
        }).T.astype(float)
        ax = comp_df.plot(kind='bar', figsize=(7, 5), color=['#d62728', '#1f77b4'])
        ax.axhline(metrics.get('min_recall_target') or 0.9, color='grey', linestyle='--', linewidth=1)
        ax.set_title('Recall vs Precision by Model (each at its own tuned threshold)')
        ax.set_ylabel('Score')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'eval_model_comparison.png', dpi=150)
        plt.close()

    logger.info(f"Evaluation plots saved to {FIGURES_DIR}")
    return metrics


if __name__ == "__main__":
    setup_logging()
    evaluate_and_save_artifacts()
