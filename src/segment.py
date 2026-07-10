# src/segment.py
"""
Segments pregnant women into clinical archetypes using K-Nearest Neighbors.

BUG FIX: the previous ARCHETYPE_CENTROIDS were defined over 8 features
including `diastolic_bp`, but nothing in the actual feature-engineering
pipeline (preprocessing.py) ever produces a `diastolic_bp` column - so
`MaternalSegmenter.predict_segments()` would raise a KeyError the first
time it was actually called on real engineered data. train.py also had
its OWN separate, slightly different 6-feature inline segmenter that it
actually used instead of this class, so the two definitions had silently
drifted apart. This file is now the single canonical segmenter (6 features,
matching what preprocessing.py actually produces) and train.py imports it
directly instead of keeping a second copy.
"""
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List

logger = logging.getLogger(__name__)

# Features order must match the scaled input and ARCHETYPE_CENTROIDS below.
SEGMENT_FEATURE_COLS: List[str] = [
    'maternal_age', 'parity', 'systolic_bp', 'hemoglobin', 'anc_visits', 'distance_to_hospital'
]

CLINICAL_ARCHETYPES = {
    0: "Archetype_A: Young, Rural, Low ANC (High Risk)",
    1: "Archetype_B: Older, Hypertensive, Urban (Moderate Risk)",
    2: "Archetype_C: Prime Age, Healthy, Good ANC (Low Risk)",
    3: "Archetype_D: High Parity, Remote, Moderate Vitals (Moderate Risk)"
}

# Centroid definitions matching SEGMENT_FEATURE_COLS above.
ARCHETYPE_CENTROIDS = np.array([
    [18, 1, 110, 10.5, 2, 45],  # Archetype A: Teen, rural, low ANC, anemic
    [36, 4, 145, 11.0, 5, 5],   # Archetype B: Older, hypertensive, urban, good ANC
    [25, 1, 115, 12.0, 8, 10],  # Archetype C: Prime age, healthy, excellent ANC
    [32, 6, 125, 11.2, 3, 35],  # Archetype D: High parity, remote, moderate ANC
])


class MaternalSegmenter:
    """
    Segments pregnant women into clinical archetypes using K-Nearest Neighbors.
    This helps the Ministry of Health allocate targeted resources (e.g., mobile clinics).
    """

    def __init__(self, feature_cols: List[str] = None):
        self.scaler = StandardScaler()
        # KNN with k=1 to assign each patient to their single closest archetype
        self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.is_fitted = False
        self.feature_cols = feature_cols or SEGMENT_FEATURE_COLS
        self.archetypes = CLINICAL_ARCHETYPES

    def fit(self, df: pd.DataFrame = None, feature_cols: List[str] = None) -> "MaternalSegmenter":
        """
        Fits the scaler and KNN model on the archetype centroids.
        `df`/`feature_cols` args are accepted for API compatibility but the
        segmenter defines its risk archetypes from clinical guidelines
        (ARCHETYPE_CENTROIDS), not from the training data distribution.
        """
        logger.info("Fitting KNN Segmenter on clinical archetypes...")
        if feature_cols is not None:
            self.feature_cols = feature_cols

        scaled_centroids = self.scaler.fit_transform(ARCHETYPE_CENTROIDS)
        self.knn.fit(scaled_centroids)
        self.is_fitted = True
        logger.info("KNN Segmenter fitted successfully.")
        return self

    def predict_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns a clinical archetype segment to each patient in the dataframe."""
        if not self.is_fitted:
            raise ValueError("Segmenter must be fitted before predicting.")

        X = df[self.feature_cols].copy()
        # .values: the scaler was fit on a plain ndarray of centroids (no feature
        # names), so passing a DataFrame here triggers a harmless-but-noisy
        # sklearn "fitted without feature names" warning. Strip the names.
        X_scaled = self.scaler.transform(X.values)

        distances, indices = self.knn.kneighbors(X_scaled)

        df = df.copy()
        df['knn_segment_id'] = indices.flatten()
        df['knn_segment_name'] = df['knn_segment_id'].map(self.archetypes)
        df['knn_distance_to_archetype'] = distances.flatten()

        logger.info(f"Assigned segments to {len(df)} patients.")
        return df

    def predict_single(self, patient_row: pd.DataFrame) -> dict:
        """Convenience method for predict.py - segments a single-row dataframe."""
        result = self.predict_segments(patient_row.iloc[[0]])
        return {
            'segment_id': int(result['knn_segment_id'].iloc[0]),
            'segment_name': result['knn_segment_name'].iloc[0],
            'distance': float(result['knn_distance_to_archetype'].iloc[0]),
        }


def save_segmenter(segmenter: MaternalSegmenter, filepath: str = 'models/knn_segmenter.pkl') -> None:
    """Saves the fitted segmenter to disk."""
    joblib.dump(segmenter, filepath)
    logger.info(f"KNN Segmenter saved to {filepath}")


def load_segmenter(filepath: str = 'models/knn_segmenter.pkl') -> MaternalSegmenter:
    """Loads the segmenter from disk."""
    segmenter = joblib.load(filepath)
    logger.info(f"KNN Segmenter loaded from {filepath}")
    return segmenter
