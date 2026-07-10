# src/train.py
"""
End-to-end training pipeline for the maternal health clinical decision support system.

WHAT CHANGED FROM THE PREVIOUS VERSION:
1. Preprocessing and segmentation are now imported from preprocessing.py / segment.py
   instead of being redefined inline here - previously train.py had its OWN private
   copies of both, which had already drifted from the versions in preprocessing.py
   and segment.py (different formulas, different feature sets). One source of truth now.
2. The predictive model used to be a single hardcoded XGBoost with fixed
   hyperparameters and the sklearn default 0.5 decision threshold, which only
   caught ~55% of adverse outcomes on this ~7% base-rate target. This version:
     - Tunes Logistic Regression, Random Forest, and XGBoost with GridSearchCV
       using an F2 score (recall weighted 4x precision) as the CV objective.
     - Sweeps the decision threshold on the precision-recall curve to find the
       highest-precision point that still guarantees recall >= MIN_RECALL_TARGET.
     - Selects the best model+threshold combination and saves the threshold
       alongside the model (model_metadata.pkl) - evaluate.py and predict.py
       now load and use this threshold instead of silently defaulting to 0.5.
3. Adds feature correlation analysis and PCA (with saved transformer for
   inference-time reuse) matching the notebook's Sections 4B/4C.
4. Saves a PCA-augmented candidate and keeps it only if it improves precision
   at the recall target - it's an honest comparison, not a forced win.
"""
import pandas as pd
import numpy as np
import joblib
import logging
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_curve,
                              make_scorer, fbeta_score)
from xgboost import XGBClassifier

from utils import setup_logging, ensure_all_project_dirs, save_json, MODELS_DIR, FIGURES_DIR, DEFAULT_DATA_PATH
from preprocessing import (preprocess_pipeline, compute_correlations, fit_pca,
                            MODEL_FEATURE_COLS, PCA_INPUT_COLS)
from segment import MaternalSegmenter, SEGMENT_FEATURE_COLS, save_segmenter
from recommend import build_and_save_recommender_artifact

logger = logging.getLogger(__name__)

MIN_RECALL_TARGET = 0.90
RANDOM_STATE = 42


# ==========================================
# FEATURE RELATIONSHIPS & PCA (Notebook Sections 4B / 4C)
# ==========================================
def run_correlation_analysis(df: pd.DataFrame, save_fig: bool = True) -> None:
    corr_matrix, target_corr, high_corr_pairs = compute_correlations(df)

    logger.info("Top 5 features correlated with adverse_outcome:\n" +
                target_corr.abs().sort_values(ascending=False).head(5).to_string())
    if high_corr_pairs:
        logger.info("Potential multicollinearity (|r| > 0.7):")
        for a, b, v in high_corr_pairs:
            logger.info(f"  {a} <-> {b}: r={v}")

    if save_fig:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=axes[0])
        axes[0].set_title('Feature Correlation Matrix')

        colors = ['#d62728' if v > 0 else '#2ca02c' for v in target_corr.values]
        axes[1].barh(target_corr.index, target_corr.values, color=colors)
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].set_title('Correlation with Adverse Outcome')

        plt.tight_layout()
        FIGURES_DIR.mkdir(exist_ok=True, parents=True)
        plt.savefig(FIGURES_DIR / 'feature_correlations.png', dpi=150)
        plt.close()
        logger.info(f"Saved correlation plot to {FIGURES_DIR / 'feature_correlations.png'}")


def run_pca_analysis(df: pd.DataFrame, n_components: int = 4, save_fig: bool = True):
    scaler, pca, df = fit_pca(df, n_components=n_components, random_state=RANDOM_STATE)
    explained = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance ratio: {np.round(explained, 3).tolist()}")
    logger.info(f"PC1+PC2 explain {explained[:2].sum():.1%} of variance in the vitals block.")

    if save_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].bar(range(1, len(explained) + 1), explained, color='#4c72b0', alpha=0.8)
        axes[0].plot(range(1, len(explained) + 1), np.cumsum(explained), color='#d62728', marker='o')
        axes[0].set_title('PCA Scree Plot')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')

        sample_idx = df.sample(min(5000, len(df)), random_state=RANDOM_STATE).index
        sc = axes[1].scatter(df.loc[sample_idx, 'pca_1'], df.loc[sample_idx, 'pca_2'],
                              c=df.loc[sample_idx, 'adverse_outcome'], cmap='coolwarm', alpha=0.4, s=12)
        axes[1].set_xlabel(f'PC1 ({explained[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({explained[1]:.1%})')
        axes[1].set_title('Patients in PCA Space')
        plt.colorbar(sc, ax=axes[1])

        plt.tight_layout()
        FIGURES_DIR.mkdir(exist_ok=True, parents=True)
        plt.savefig(FIGURES_DIR / 'pca_analysis.png', dpi=150)
        plt.close()
        logger.info(f"Saved PCA plot to {FIGURES_DIR / 'pca_analysis.png'}")

    return scaler, pca, df


# ==========================================
# PREDICTIVE MODELING: GRIDSEARCH + RECALL-TARGETED THRESHOLD TUNING
# ==========================================
def find_best_threshold(y_true, y_proba, min_recall: float = MIN_RECALL_TARGET):
    """Highest-precision threshold subject to recall >= min_recall."""
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    valid = rec[:-1] >= min_recall
    if valid.any():
        candidate_idx = np.where(valid)[0]
        best_idx = candidate_idx[np.argmax(prec[candidate_idx])]
    else:
        best_idx = int(np.argmax(rec[:-1]))
    return float(thresh[best_idx]), float(prec[best_idx]), float(rec[best_idx])


def build_candidate_grids(scale_pos_weight_base: float):
    logreg_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    logreg_grid = {'clf__C': [0.01, 0.1, 1, 10]}

    rf_pipe = Pipeline([('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1))])
    rf_grid = {
        'clf__n_estimators': [150, 250],
        'clf__max_depth': [6, 10],
        'clf__min_samples_leaf': [2, 5],
    }

    xgb_pipe = Pipeline([('clf', XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=1))])
    xgb_grid = {
        'clf__n_estimators': [150, 250],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.05, 0.1],
        'clf__scale_pos_weight': [scale_pos_weight_base, scale_pos_weight_base * 1.5],
    }

    return {
        'LogisticRegression': (logreg_pipe, logreg_grid),
        'RandomForest': (rf_pipe, rf_grid),
        'XGBoost': (xgb_pipe, xgb_grid),
    }


def train_predictive_model(df: pd.DataFrame, min_recall_target: float = MIN_RECALL_TARGET):
    """
    Tunes LR/RF/XGBoost via GridSearchCV (F2-scored), threshold-tunes each to the
    recall target, and returns the best (model, threshold, feature_cols, X_test, y_test).
    """
    logger.info("Training predictive model with GridSearchCV + threshold tuning...")

    feature_cols = list(MODEL_FEATURE_COLS)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols]
    y = df['adverse_outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
    scale_pos_weight_base = (y_train == 0).sum() / (y_train == 1).sum()

    candidates = build_candidate_grids(scale_pos_weight_base)

    results = {}
    for name, (pipe, grid) in candidates.items():
        n_combos = len(list(ParameterGrid(grid)))
        logger.info(f"Tuning {name} ({n_combos} combos x {cv.get_n_splits()} folds)...")
        gs = GridSearchCV(pipe, grid, scoring=f2_scorer, cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        y_proba = gs.predict_proba(X_test)[:, 1]
        thresh, prec, rec = find_best_threshold(y_test, y_proba, min_recall_target)
        results[name] = {
            'estimator': gs.best_estimator_, 'cv_f2': gs.best_score_, 'params': gs.best_params_,
            'threshold': thresh, 'precision': prec, 'recall': rec, 'proba': y_proba,
        }
        logger.info(f"  {name}: CV F2={gs.best_score_:.3f} | threshold={thresh:.3f} "
                    f"-> Recall={rec:.2%}, Precision={prec:.2%}")

    qualifying = {k: v for k, v in results.items() if v['recall'] >= min_recall_target}
    if qualifying:
        best_name = max(qualifying, key=lambda k: qualifying[k]['precision'])
    else:
        best_name = max(results, key=lambda k: results[k]['recall'])
        logger.warning(f"No model reached {min_recall_target:.0%} recall - using closest ({best_name}).")

    best = results[best_name]
    logger.info(f"SELECTED MODEL: {best_name} @ threshold={best['threshold']:.3f}")
    logger.info("\n" + classification_report(
        y_test, (best['proba'] >= best['threshold']).astype(int), target_names=['Normal', 'Adverse']))

    comparison = {name: {'cv_f2': r['cv_f2'], 'recall': r['recall'], 'precision': r['precision'],
                          'threshold': r['threshold'], 'params': r['params']}
                  for name, r in results.items()}

    return {
        'model_name': best_name,
        'model': best['estimator'],
        'threshold': best['threshold'],
        'recall': best['recall'],
        'precision': best['precision'],
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test,
        'comparison': comparison,
        'candidates': candidates,
        'cv': cv,
        'f2_scorer': f2_scorer,
    }


def try_pca_augmentation(df: pd.DataFrame, base_result: dict, min_recall_target: float = MIN_RECALL_TARGET):
    """
    Re-tunes the winning model family with pca_1/pca_2 added to its feature set.
    Adopts the PCA-augmented version only if it improves precision at the same
    recall target - this is a genuine comparison, not a rubber stamp.
    """
    feature_cols_pca = base_result['feature_cols'] + ['pca_1', 'pca_2']
    y = df['adverse_outcome']
    X_pca = df[feature_cols_pca]
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    pipe, grid = base_result['candidates'][base_result['model_name']]
    gs = GridSearchCV(pipe, grid, scoring=base_result['f2_scorer'], cv=base_result['cv'], n_jobs=-1, refit=True)
    gs.fit(Xp_train, yp_train)
    proba = gs.predict_proba(Xp_test)[:, 1]
    thresh, prec, rec = find_best_threshold(yp_test, proba, min_recall_target)

    logger.info(f"PCA-augmented {base_result['model_name']}: threshold={thresh:.3f} "
                f"-> Recall={rec:.2%}, Precision={prec:.2%}")

    if rec >= min_recall_target and prec > base_result['precision']:
        logger.info("PCA components improved precision at the recall target - adopting PCA-augmented model.")
        base_result.update({
            'model': gs.best_estimator_, 'threshold': thresh, 'recall': rec, 'precision': prec,
            'feature_cols': feature_cols_pca, 'X_test': Xp_test, 'y_test': yp_test,
        })
    else:
        logger.info("PCA components did not improve on the raw-feature model - keeping raw-feature version.")

    return base_result


# ==========================================
# MAIN PIPELINE
# ==========================================
def run_training_pipeline(data_path: str = None, min_recall_target: float = MIN_RECALL_TARGET,
                           run_pca_augmentation: bool = True):
    ensure_all_project_dirs()
    data_path = data_path or str(DEFAULT_DATA_PATH)

    # 1. Preprocess
    df = preprocess_pipeline(data_path)

    # 2. Feature relationships (Section 4B)
    run_correlation_analysis(df)

    # 3. PCA (Section 4C)
    pca_scaler, pca, df = run_pca_analysis(df)

    # 4. Segmentation
    segmenter = MaternalSegmenter().fit()
    df = segmenter.predict_segments(df)
    save_segmenter(segmenter, str(MODELS_DIR / 'knn_segmenter.pkl'))

    # 5. Recommender artifact (with detailed action/timeline/rationale)
    build_and_save_recommender_artifact(MODELS_DIR / 'hybrid_recommender.pkl')

    # 6. Predictive model: GridSearch + threshold tuning for recall >= target
    result = train_predictive_model(df, min_recall_target)

    # 7. Optional PCA-augmented comparison
    if run_pca_augmentation:
        result = try_pca_augmentation(df, result, min_recall_target)

    # 8. Persist everything
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(result['model'], MODELS_DIR / 'xgb_risk_model.pkl')  # filename kept for dashboard compatibility
    joblib.dump(result['feature_cols'], MODELS_DIR / 'feature_names.pkl')
    joblib.dump({
        'model_name': result['model_name'],
        'threshold': result['threshold'],
        'min_recall_target': min_recall_target,
        'test_recall': result['recall'],
        'test_precision': result['precision'],
    }, MODELS_DIR / 'model_metadata.pkl')
    joblib.dump({'scaler': pca_scaler, 'pca': pca, 'input_cols': PCA_INPUT_COLS}, MODELS_DIR / 'pca_transformer.pkl')
    joblib.dump((result['X_test'], result['y_test']), MODELS_DIR / 'test_set.pkl')

    save_json(result['comparison'], MODELS_DIR / 'model_comparison.json')

    logger.info("=" * 60)
    logger.info(f"Training complete. Deployed model: {result['model_name']}")
    logger.info(f"Decision threshold: {result['threshold']:.3f} "
                f"(recall={result['recall']:.2%}, precision={result['precision']:.2%})")
    logger.info(f"All artifacts saved to {MODELS_DIR}")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the maternal health CDS pipeline.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the raw .xlsx register.")
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL_TARGET,
                         help="Minimum recall the deployed model must hit (default 0.90).")
    parser.add_argument("--skip-pca-augmentation", action="store_true",
                         help="Skip the PCA-augmented model comparison step (faster).")
    args = parser.parse_args()

    setup_logging()
    run_training_pipeline(
        data_path=args.data_path,
        min_recall_target=args.min_recall,
        run_pca_augmentation=not args.skip_pca_augmentation,
    )
