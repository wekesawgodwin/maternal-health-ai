# src/recommend.py
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / 'models' / 'hybrid_recommender.pkl'

class HybridClinicalRecommender:
    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT_PATH):
        self.artifact_path = artifact_path
        self.is_loaded = False
        
    def load_artifact(self) -> None:
        if self.is_loaded: return
        artifact = joblib.load(self.artifact_path)
        self.interventions = artifact['interventions']
        self.content_matrix = artifact['content_matrix']
        self.collaborative_matrix = artifact['collaborative_matrix']
        self.feature_cols = artifact['feature_cols']
        self.is_loaded = True

    def get_recommendations(self, patient_df: pd.DataFrame, alpha: float = 0.7, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_loaded: self.load_artifact()
            
        # Extract the 7 binary clinical flags
        patient_flags = patient_df[self.feature_cols].values.reshape(1, -1)
        
        # 1. Content Score: Cosine similarity between patient flags and intervention requirements
        # Because both are binary (0/1), this acts as a perfect clinical rule matcher
        content_scores = cosine_similarity(patient_flags, self.content_matrix)[0]
        
        # 2. Collaborative Score: Historical success based on KNN segment
        segment_id = int(patient_df['knn_segment_id'].iloc[0])
        collab_scores = self.collaborative_matrix[segment_id]
        
        # 3. Hybrid Score
        hybrid_scores = (alpha * content_scores) + ((1 - alpha) * collab_scores)
        top_indices = hybrid_scores.argsort()[::-1][:top_k]
        
        recommendations = []
        for i in top_indices:
            recommendations.append({
                'intervention': self.interventions[i].replace('_', ' '),
                'hybrid_score': round(float(hybrid_scores[i]), 3),
                'content_contrib': round(float(content_scores[i]), 3),
                'collab_contrib': round(float(collab_scores[i]), 3)
            })
        return recommendations

def recommend_interventions(patient_df: pd.DataFrame, alpha: float = 0.7) -> List[Dict[str, Any]]:
    recommender = HybridClinicalRecommender()
    recommender.load_artifact()
    return recommender.get_recommendations(patient_df, alpha=alpha)