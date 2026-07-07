# src/preprocessing.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads the raw maternity register data and performs initial cleaning."""
    logging.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    
    # Filter for Kenya (country == 2) to align with local KHIS context
    df = df[df['country'] == 2].copy()
    
    # Drop duplicates based on unique maternal record ID
    df.drop_duplicates(subset=['id'], inplace=True)
    
    # Filter for actual births (exclude pure abortions/discharges without birth for this model)
    df = df[df['record_type'] == 'Birth'].copy()
    
    logging.info(f"Data cleaned. Shape: {df.shape}")
    return df

def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """Defines the adverse outcome target variable."""
    adverse_conditions = [
        'Fresh_Still_Birth', 'Macerated_Still_Birth', 
        'Immediate_Neonatal_Death'
    ]
    df['adverse_outcome'] = (
        (df['c_mother_status'] == 'Died') | 
        (df['c_baby_status'].isin(adverse_conditions))
    ).astype(int)
    return df

def engineer_anc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers clinically realistic ANC features. 
    In a real deployment, these would come from the ANC register (KHIS).
    Here, we simulate them based on demographic correlations.
    """
    np.random.seed(42)
    n = len(df)
    
    # Map facility codes to Kenya Facility Levels (Simulated)
    facility_levels = {f'HF{str(i).zfill(3)}': np.random.choice([2, 3, 4, 5]) for i in range(1, 24)}
    df['facility_level'] = df['facility_coded'].map(facility_levels)
    
    # Simulate ANC variables with realistic distributions
    # Higher age and parity correlate with higher BP and lower ANC visits
    age_numeric = df['mothers_age_cat'].map({'≤19': 17, '20-24': 22, '25-29': 27, '30-34': 32, '≥35': 38, 'missing': 25})
    
    df['maternal_age'] = age_numeric
    df['parity'] = np.random.poisson(lam=2, size=n) # Simulated parity
    
    # Systolic BP: Base 110, increases with age
    df['systolic_bp'] = 110 + (df['maternal_age'] - 20) * 0.5 + np.random.normal(0, 10, n)
    df['diastolic_bp'] = 70 + (df['maternal_age'] - 20) * 0.3 + np.random.normal(0, 6, n)
    
    # BMI: Base 22, slight increase with age
    df['maternal_bmi'] = 22 + (df['maternal_age'] - 20) * 0.1 + np.random.normal(0, 3, n)
    
    # Hemoglobin: Base 11.5, lower in younger/older mothers
    df['hemoglobin'] = 11.5 - np.abs(df['maternal_age'] - 27) * 0.05 + np.random.normal(0, 1.2, n)
    
    # ANC Visits: WHO recommends minimum 8. Lower in rural (Level 2) facilities
    base_anc = np.where(df['facility_level'] <= 2, 3, 5)
    df['anc_visits'] = np.clip(base_anc + np.random.normal(0, 1.5, n), 0, 10).astype(int)
    
    # Distance to referral hospital (km)
    df['distance_to_hospital'] = np.where(
        df['facility_level'] >= 4, 
        np.random.exponential(5, n), 
        np.random.exponential(25, n)
    )
    
    # Clinical Flags
    df['hypertension_flag'] = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int)
    df['anemia_flag'] = (df['hemoglobin'] < 11.0).astype(int)
    df['low_anc_flag'] = (df['anc_visits'] < 4).astype(int)
    df['high_bmi_flag'] = (df['maternal_bmi'] >= 30).astype(int)
    
    logging.info("Feature engineering completed.")
    return df

def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    """Main preprocessing orchestration."""
    df = load_and_clean_data(filepath)
    df = define_target(df)
    df = engineer_anc_features(df)
    return df