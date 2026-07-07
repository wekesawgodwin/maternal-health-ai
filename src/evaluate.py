# src/evaluate.py
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             average_precision_score, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

def evaluate_and_save_artifacts():
    logger.info("Loading models and test data...")
    xgb_model = joblib.load(MODELS_DIR / 'xgb_risk_model.pkl')
    X_test, y_test = joblib.load(MODELS_DIR / 'test_set.pkl')
    
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # ==========================================
    # 1. CALCULATE & SAVE METRICS (JSON)
    # ==========================================
    report = classification_report(y_test, y_pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        "recall": round(report['1']['recall'], 4),
        "precision": round(report['1']['precision'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn)
    }
    
    with open(MODELS_DIR / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved to models/evaluation_metrics.json")
    
    # ==========================================
    # 2. GENERATE & SAVE INDIVIDUAL PLOTS (PNG)
    # ==========================================
    logger.info("Generating evaluation plots...")
    
    # A. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Adverse'], yticklabels=['Normal', 'Adverse'])
    plt.title('Confusion Matrix')
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
    
    # C. Precision-Recall Curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prec_curve, color='green', lw=2, label=f'PR curve (AP = {metrics["pr_auc"]:.2f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
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
    
    logger.info("Evaluation plots saved to /figures directory.")

if __name__ == "__main__":
    evaluate_and_save_artifacts()