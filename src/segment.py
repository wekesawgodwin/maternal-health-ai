# src/segment.py
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# Define Clinical Archetypes (Centroids) based on Kenyan MoH risk profiles
# Features order must match the scaled input: 
# [maternal_age, parity, systolic_bp, diastolic_bp, maternal_bmi, hemoglobin, anc_visits, distance_to_hospital]
CLINICAL_ARCHETYPES = {
    0: "Archetype_A: Young, Rural, Low ANC (High Risk)",
    1: "Archetype_B: Older, Comorbid, Urban (Moderate Risk)",
    2: "Archetype_C: Prime Age, Normal Vitals, Good ANC (Low Risk)",
    3: "Archetype_D: High Parity, Remote, Moderate Vitals (Moderate Risk)"
}

# Centroid definitions matching the features above
ARCHETYPE_CENTROIDS = np.array([
    [18, 1, 110, 70, 20, 10.5, 2, 45],  # Archetype A: Teen, rural, low ANC, anemic
    [36, 4, 145, 90, 28, 11.0, 5, 5],   # Archetype B: Older, hypertensive, urban, good ANC
    [25, 1, 115, 75, 23, 12.0, 8, 10],  # Archetype C: Prime age, healthy, excellent ANC
    [32, 6, 125, 80, 24, 11.2, 3, 35]   # Archetype D: High parity, remote, moderate ANC
])

class MaternalSegmenter:
    """
    Segments pregnant women into clinical archetypes using K-Nearest Neighbors.
    This helps the Ministry of Health allocate targeted resources (e.g., mobile clinics).
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        # KNN with k=1 to assign each patient to their single closest archetype
        self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, feature_cols: list) -> None:
        """Fits the scaler and KNN model on the archetype centroids."""
        logger.info("Fitting KNN Segmenter on clinical archetypes...")
        
        # Scale the centroids using the same scaler logic (we fit scaler on centroids to define the space)
        scaled_centroids = self.scaler.fit_transform(ARCHETYPE_CENTROIDS)
        self.knn.fit(scaled_centroids)
        self.feature_cols = feature_cols
        self.is_fitted = True
        logger.info("KNN Segmenter fitted successfully.")
        
    def predict_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns a clinical archetype segment to each patient in the dataframe."""
        if not self.is_fitted:
            raise ValueError("Segmenter must be fitted before predicting.")
            
        # Ensure we only use the required features
        X = df[self.feature_cols].copy()
        
        # Scale the patient data
        X_scaled = self.scaler.transform(X)
        
        # Find the nearest archetype for each patient
        distances, indices = self.knn.kneighbors(X_scaled)
        
        # Map indices to archetype names
        df['knn_segment_id'] = indices.flatten()
        df['knn_segment_name'] = df['knn_segment_id'].map(CLINICAL_ARCHETYPES)
        df['knn_distance_to_archetype'] = distances.flatten()
        
        logger.info(f"Assigned segments to {len(df)} patients.")
        return df

def save_segmenter(segmenter: MaternalSegmenter, filepath: str = 'models/knn_segmenter.pkl') -> None:
    """Saves the fitted segmenter to disk."""
    joblib.dump(segmenter, filepath)
    logger.info(f"KNN Segmenter saved to {filepath}")

def load_segmenter(filepath: str = 'models/knn_segmenter.pkl') -> MaternalSegmenter:
    """Loads the segmenter from disk."""
    segmenter = joblib.load(filepath)
    logger.info(f"KNN Segmenter loaded from {filepath}")
    return segmenter