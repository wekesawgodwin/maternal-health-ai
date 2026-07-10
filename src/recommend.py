# src/recommend.py
"""
Hybrid content + collaborative recommender for clinical interventions.

IMPROVEMENT OVER PREVIOUS VERSION: `get_recommendations` used to return only
a bare intervention name and score - a nurse reading it would have no idea
*what to actually do*, *when*, or *why the system suggested it*. This now
matches the notebook's detailed report: every recommendation carries a
concrete action, a timeline, and a plain-language clinical rationale, and
`generate_report` layers the model's own risk probability + urgency tier
(LOW/ELEVATED/HIGH/CRITICAL) on top, mirroring
`generate_detailed_recommendation()` from the notebook.
"""
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

from utils import MODELS_DIR

logger = logging.getLogger(__name__)
DEFAULT_ARTIFACT_PATH = MODELS_DIR / 'hybrid_recommender.pkl'

INTERVENTIONS = ['Routine_Care', 'Nutrition_Support', 'Hypertension_Mgmt',
                  'Anemia_Mgmt', 'Level4_Referral', 'Emergency_Transport']

# Concrete action, timeline, and plain-language rationale for each intervention.
INTERVENTION_DETAILS = {
    'Routine_Care': {
        'action': 'Continue the standard ANC visit schedule per MoH guidelines.',
        'timeline': 'Next routine ANC visit',
        'rationale': 'No active risk flags detected.'
    },
    'Nutrition_Support': {
        'action': 'Enroll in iron/folate supplementation and nutrition counseling.',
        'timeline': 'Within 1 week',
        'rationale': 'Low hemoglobin and/or insufficient ANC attendance detected.'
    },
    'Hypertension_Mgmt': {
        'action': 'Start BP monitoring protocol; consider antihypertensive per facility guidelines.',
        'timeline': 'Immediate - within 24-48 hours',
        'rationale': 'Systolic BP >= 140 mmHg (hypertension flag positive).'
    },
    'Anemia_Mgmt': {
        'action': 'Confirm with a lab hemoglobin test; start iron therapy.',
        'timeline': 'Within 3-5 days',
        'rationale': 'Hemoglobin below the 11.0 g/dL threshold.'
    },
    'Level4_Referral': {
        'action': 'Refer to a Level 4+ facility with EmONC capability.',
        'timeline': 'Immediate',
        'rationale': 'Combination of preterm birth, low birth weight, or prior referral history.'
    },
    'Emergency_Transport': {
        'action': 'Arrange emergency transport (ambulance/voucher) to the nearest capable facility.',
        'timeline': 'Immediate - do not delay',
        'rationale': 'High distance to hospital and/or referred-in status increases delay risk.'
    },
}

# Content Matrix: MoH Guidelines (Binary Flags: HTN, Anemia, Preterm, LBW, Low ANC, High Dist, Referral)
CONTENT_MATRIX = np.array([
    [0, 0, 0, 0, 0, 0, 0],  # Routine_Care
    [0, 1, 0, 0, 1, 0, 0],  # Nutrition_Support
    [1, 0, 0, 0, 0, 0, 0],  # Hypertension_Mgmt
    [0, 1, 0, 0, 0, 0, 0],  # Anemia_Mgmt
    [1, 0, 1, 1, 0, 0, 1],  # Level4_Referral
    [0, 0, 0, 0, 1, 1, 0],  # Emergency_Transport
])

# Collaborative Matrix: Historical Success by Archetype
COLLABORATIVE_MATRIX = np.array([
    [0.4, 0.5, 0.6, 0.5, 0.8, 0.9],  # Archetype A
    [0.6, 0.6, 0.9, 0.6, 0.7, 0.5],  # Archetype B
    [0.9, 0.8, 0.5, 0.8, 0.4, 0.3],  # Archetype C
    [0.5, 0.6, 0.7, 0.6, 0.9, 0.8],  # Archetype D
])

FEATURE_COLS = ['hypertension_flag', 'anemia_flag', 'preterm_flag',
                 'low_bw_flag', 'low_anc_flag', 'high_distance_flag', 'referral_in']

# Human-readable flag labels used in the report, populated with the patient's
# actual measured value (e.g. "Hypertension (Systolic BP 160 mmHg)").
FLAG_LABEL_TEMPLATES = {
    'hypertension_flag': lambda row: f"Hypertension (Systolic BP {row['systolic_bp']:.0f} mmHg)",
    'anemia_flag': lambda row: f"Anemia (Hemoglobin {row['hemoglobin']:.1f} g/dL)",
    'preterm_flag': lambda row: f"Preterm gestation ({row.get('gestational_age_wks', float('nan')):.0f} wks)",
    'low_bw_flag': lambda row: f"Low birth weight ({row.get('birth_weight_g', float('nan')):.0f} g)",
    'low_anc_flag': lambda row: f"Insufficient ANC visits ({row['anc_visits']:.0f} visits)",
    'high_distance_flag': lambda row: f"High distance to facility ({row['distance_to_hospital']:.0f} km)",
    'referral_in': lambda row: "Referred-in case",
}


def build_and_save_recommender_artifact(filepath: Path = DEFAULT_ARTIFACT_PATH) -> dict:
    """Creates and saves the static matrices + detail text for the Hybrid Recommender."""
    logger.info("Building Clinical Flag-Based Recommender...")
    artifact = {
        'interventions': INTERVENTIONS,
        'content_matrix': CONTENT_MATRIX,
        'collaborative_matrix': COLLABORATIVE_MATRIX,
        'feature_cols': FEATURE_COLS,
        'intervention_details': INTERVENTION_DETAILS,
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, filepath)
    logger.info(f"Hybrid Recommender artifact saved to {filepath}")
    return artifact


def risk_tier(probability: float, threshold: float) -> str:
    """Maps a model probability to a clinical urgency tier relative to the deployed threshold."""
    if probability >= max(threshold * 2, 0.6):
        return "CRITICAL"
    elif probability >= threshold:
        return "HIGH"
    elif probability >= threshold * 0.5:
        return "ELEVATED"
    return "LOW"


class HybridClinicalRecommender:
    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT_PATH):
        self.artifact_path = Path(artifact_path)
        self.is_loaded = False

    def load_artifact(self) -> None:
        if self.is_loaded:
            return
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"No recommender artifact at {self.artifact_path}. Run train.py first."
            )
        artifact = joblib.load(self.artifact_path)
        self.interventions = artifact['interventions']
        self.content_matrix = artifact['content_matrix']
        self.collaborative_matrix = artifact['collaborative_matrix']
        self.feature_cols = artifact['feature_cols']
        # Older artifacts (saved before this fix) won't have intervention_details -
        # fall back to the in-code copy so old model dirs don't hard-crash.
        self.intervention_details = artifact.get('intervention_details', INTERVENTION_DETAILS)
        self.is_loaded = True

    def get_recommendations(self, patient_df: pd.DataFrame, alpha: float = 0.7, top_k: int = 3) -> List[Dict[str, Any]]:
        """Returns ranked interventions with scores AND concrete action/timeline/rationale."""
        if not self.is_loaded:
            self.load_artifact()

        patient_flags = patient_df[self.feature_cols].values.reshape(1, -1)
        content_scores = cosine_similarity(patient_flags, self.content_matrix)[0]

        segment_id = int(patient_df['knn_segment_id'].iloc[0])
        collab_scores = self.collaborative_matrix[segment_id]

        hybrid_scores = (alpha * content_scores) + ((1 - alpha) * collab_scores)
        top_indices = hybrid_scores.argsort()[::-1][:top_k]

        recommendations = []
        for i in top_indices:
            name = self.interventions[i]
            detail = self.intervention_details.get(name, {})
            recommendations.append({
                'intervention': name.replace('_', ' '),
                'hybrid_score': round(float(hybrid_scores[i]), 3),
                'confidence': f"{hybrid_scores[i] * 100:.1f}%",
                'content_contrib': round(float(content_scores[i]), 3),
                'collab_contrib': round(float(collab_scores[i]), 3),
                'action': detail.get('action', 'See clinical guidelines.'),
                'timeline': detail.get('timeline', 'As soon as possible'),
                'rationale': detail.get('rationale', ''),
            })
        return recommendations

    def generate_report(
        self,
        patient_df: pd.DataFrame,
        risk_probability: Optional[float] = None,
        threshold: float = 0.5,
        alpha: float = 0.7,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Full clinical decision-support report for a single patient row:
        risk probability + tier, active flags with plain-language rationale,
        cohort archetype, and detailed top-K interventions. Mirrors the
        notebook's generate_detailed_recommendation().
        """
        row = patient_df.iloc[0]
        report: Dict[str, Any] = {}

        if risk_probability is not None:
            report['risk_probability'] = round(float(risk_probability), 3)
            report['risk_tier'] = risk_tier(risk_probability, threshold)
        else:
            report['risk_probability'] = None
            report['risk_tier'] = None

        active_flags = []
        for flag in self.feature_cols if self.is_loaded else FEATURE_COLS:
            if int(row.get(flag, 0)) == 1:
                label_fn = FLAG_LABEL_TEMPLATES.get(flag)
                active_flags.append(label_fn(row) if label_fn else flag)
        report['active_flags'] = active_flags if active_flags else ["No active clinical risk flags"]

        report['archetype'] = row.get('knn_segment_name', 'Unknown')
        report['recommendations'] = self.get_recommendations(patient_df, alpha=alpha, top_k=top_k)
        return report


def recommend_interventions(patient_df: pd.DataFrame, alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Backwards-compatible functional entry point."""
    recommender = HybridClinicalRecommender()
    recommender.load_artifact()
    return recommender.get_recommendations(patient_df, alpha=alpha)


def print_recommendation_report(report: Dict[str, Any], patient_label: str = "Patient") -> None:
    """Pretty-prints a report dict from generate_report() - handy for CLI/debugging."""
    print(f"\n{'=' * 70}")
    print(f"CLINICAL DECISION SUPPORT REPORT - {patient_label}")
    print(f"{'=' * 70}")
    if report.get('risk_probability') is not None:
        print(f"Model Risk Probability : {report['risk_probability'] * 100:.1f}%")
        print(f"Risk Tier              : {report['risk_tier']}")
    print(f"Patient Archetype       : {report['archetype']}")
    print("Active Clinical Flags   :")
    for f in report['active_flags']:
        print(f"   - {f}")
    print(f"\nTop {len(report['recommendations'])} Recommended Interventions:")
    for i, r in enumerate(report['recommendations'], 1):
        print(f"  {i}. {r['intervention']}  (confidence: {r['confidence']})")
        print(f"     Action    : {r['action']}")
        print(f"     Timeline  : {r['timeline']}")
        print(f"     Rationale : {r['rationale']}")
    print(f"{'=' * 70}\n")
