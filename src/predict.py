# src/predict.py
"""
Single-patient inference orchestration.

CRITICAL FIX: the previous version loaded 'models/best_model.pkl' and
'models/preprocessor.pkl' - neither file is ever created by train.py (which
saves 'xgb_risk_model.pkl' and doesn't fit a separate preprocessor at all,
since the model consumes raw engineered features directly). This module
would have raised FileNotFoundError on the very first call. It also used
hardcoded risk-category cutoffs (0.15 / 0.45) that had no relationship to
the model's actual tuned decision threshold.

This version:
  - Loads the artifacts train.py actually saves (model, feature list,
    metadata with the tuned threshold, PCA transformer, segmenter, recommender).
  - Accepts raw clinical inputs (the fields a nurse would actually enter),
    derives the same binary flags used at training time, and - only if the
    deployed model needs them - applies the fitted PCA transform.
  - Uses the tuned threshold (not 0.5, not hardcoded cutoffs) for the
    ADVERSE/NORMAL classification, and layers the risk-tier + recommender
    report on top, mirroring the notebook's full patient report.
"""
import logging
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from utils import setup_logging, load_model_metadata, MODELS_DIR
from preprocessing import apply_pca
from segment import MaternalSegmenter, load_segmenter
from recommend import HybridClinicalRecommender

logger = logging.getLogger(__name__)


class MaternalRiskPredictor:
    """Loads all trained artifacts once and serves full patient reports."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self._loaded = False

    def load(self) -> "MaternalRiskPredictor":
        if self._loaded:
            return self
        logger.info(f"Loading artifacts from {self.models_dir}...")
        self.model = joblib.load(self.models_dir / 'xgb_risk_model.pkl')
        self.feature_cols = joblib.load(self.models_dir / 'feature_names.pkl')
        self.metadata = load_model_metadata(self.models_dir)
        self.threshold = self.metadata['threshold']

        pca_path = self.models_dir / 'pca_transformer.pkl'
        self.pca_artifact = joblib.load(pca_path) if pca_path.exists() else None

        self.segmenter: MaternalSegmenter = load_segmenter(str(self.models_dir / 'knn_segmenter.pkl'))
        self.recommender = HybridClinicalRecommender(self.models_dir / 'hybrid_recommender.pkl')
        self.recommender.load_artifact()

        self._loaded = True
        logger.info(f"Model: {self.metadata.get('model_name')} | threshold: {self.threshold:.3f}")
        return self

    @staticmethod
    def build_patient_frame(patient_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Derives the binary clinical flags from raw patient measurements, the
        same way engineer_anc_features() does in preprocessing.py. Expected
        keys in patient_input: maternal_age, parity, systolic_bp, hemoglobin,
        anc_visits, distance_to_hospital, referral_in, maternal_bmi (optional,
        only needed for PCA), gestational_age_wks (optional), birth_weight_g (optional).
        """
        row = dict(patient_input)  # shallow copy
        row.setdefault('referral_in', 0)
        row.setdefault('gestational_age_wks', 39)
        row.setdefault('birth_weight_g', 3250)
        row.setdefault('maternal_bmi', 22.0)

        row['hypertension_flag'] = int(row['systolic_bp'] >= 140)
        row['anemia_flag'] = int(row['hemoglobin'] < 11.0)
        row['preterm_flag'] = int(row['gestational_age_wks'] < 37)
        row['low_bw_flag'] = int(row['birth_weight_g'] < 2500)
        row['low_anc_flag'] = int(row['anc_visits'] < 4)
        row['high_distance_flag'] = int(row['distance_to_hospital'] > 20)

        return pd.DataFrame([row])

    def predict(self, patient_input: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """Returns a full clinical decision-support report for one patient."""
        if not self._loaded:
            self.load()

        df = self.build_patient_frame(patient_input)

        # Apply PCA only if the deployed model was trained on PCA-augmented features.
        if self.pca_artifact is not None and any(c.startswith('pca_') for c in self.feature_cols):
            df = apply_pca(df, self.pca_artifact['scaler'], self.pca_artifact['pca'])

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required model features in patient_input: {missing}")

        probability = float(self.model.predict_proba(df[self.feature_cols])[0, 1])
        is_adverse_risk = probability >= self.threshold

        segment = self.segmenter.predict_single(df)
        df['knn_segment_id'] = segment['segment_id']
        df['knn_segment_name'] = segment['segment_name']

        report = self.recommender.generate_report(
            df, risk_probability=probability, threshold=self.threshold, top_k=top_k
        )
        report['flagged_adverse_risk'] = bool(is_adverse_risk)
        report['model_name'] = self.metadata.get('model_name')
        report['decision_threshold'] = self.threshold
        return report


# ---- Backwards-compatible functional API ----
_predictor: Optional[MaternalRiskPredictor] = None


def predict_patient_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Functional entry point kept for compatibility with older call sites.
    `patient_data` is a dict of raw clinical measurements (see
    MaternalRiskPredictor.build_patient_frame for expected keys).
    """
    global _predictor
    if _predictor is None:
        _predictor = MaternalRiskPredictor().load()
    return _predictor.predict(patient_data)


if __name__ == "__main__":
    setup_logging()
    from recommend import print_recommendation_report

    predictor = MaternalRiskPredictor().load()

    sample_high_risk = {
        'maternal_age': 19, 'parity': 1, 'systolic_bp': 160, 'hemoglobin': 8.2,
        'anc_visits': 1, 'distance_to_hospital': 43, 'referral_in': 1,
        'gestational_age_wks': 33, 'birth_weight_g': 2100, 'maternal_bmi': 21,
    }
    report = predictor.predict(sample_high_risk)
    print_recommendation_report(report, patient_label="Sample High-Risk Patient")
