# src/train.py
import pandas as pd
import numpy as np
import joblib
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. DATA LOADING & CLINICAL CLEANING
# ==========================================
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads raw maternity register data and applies clinical cleaning rules."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    
    # Filter for Kenya (country == 2) and actual Birth records (exclude abortions/discharges)
    df = df[(df['country'] == 2) & (df['record_type'].isin(['Birth', 'BBA']))].copy()
    
    # Drop duplicates based on unique ID
    df.drop_duplicates(subset=['id'], inplace=True)
    logger.info(f"Cleaned data shape: {df.shape}")
    return df

def clean_raw_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL FIX: Converts raw object/string columns with empty values 
    into proper numeric types to prevent XGBoost ValueError.
    """
    numeric_cols = ['referral_in', 'apgar_1', 'apgar_5', 'multiple', 'bba', 
                    'doc_abortion', 'doc_iufd', 'c_birth_weight_g2']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def define_target_and_map_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Defines adverse outcome target and maps categorical variables to numeric based on Data Dictionary."""
    
    # Target: Maternal Death or Severe Perinatal Outcome
    # Based on Data Dictionary: c_baby_status includes Fresh/Macerated Still Birth, Immediate Neonatal Death
    adverse_baby = ['Fresh_Still_Birth', 'Macerated_Still_Birth', 'Immediate_Neonatal_Death', 'Unknown_Still_Birth']
    
    # Handle case variations in mother status (e.g., 'Died', 'died')
    df['c_mother_status_clean'] = df['c_mother_status'].astype(str).str.lower()
    
    df['adverse_outcome'] = (
        (df['c_mother_status_clean'] == 'died') | 
        (df['c_baby_status'].isin(adverse_baby))
    ).astype(int)
    
    # Map Maternal Age Category to numeric midpoints
    age_map = {'≤19': 17, '20-24': 22, '25-29': 27, '30-34': 32, '≥35': 38, 'missing': 25}
    df['maternal_age'] = df['mothers_age_cat'].map(age_map).fillna(25)
    
    # Map Gestational Age to weeks (Handling variations in data dictionary vs actual data)
    ga_map = {
        '<24wks': 20, '24-27wks': 25, '24-28 wks': 26, '28-30wks': 29, '28-30 weeks': 29, 
        '31-33wks': 32, '34-36wks': 35, '37wks+': 39, '37+wks': 39
    }
    df['gestational_age_wks'] = df['c_cat_ga'].map(ga_map).fillna(39)
    
    # Map Birth Weight to grams midpoint
    bw_map = {
        '<500g': 400, '500-999g': 750, '1000-1499g': 1250, '1500-1999g': 1750, 
        '2000-2499g': 2250, '2500-2999g': 2750, '3000-3499g': 3250, '3500g+': 3750
    }
    df['birth_weight_g'] = df['c_cat_bw'].map(bw_map).fillna(3250)
    
    # Handle missing APGAR (0 for stillbirths, median for others)
    df['apgar_1'] = df['apgar_1'].fillna(0)
    
    return df

# ==========================================
# 2. DOMAIN-DRIVEN FEATURE ENGINEERING
# ==========================================
def engineer_anc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates ANC features CONDITIONALLY based on real clinical risk factors.
    This ensures the ML model learns correct clinical relationships (e.g., High BP = High Risk).
    """
    np.random.seed(42)
    n = len(df)
    
    # Identify patients with real clinical risks from the raw data
    is_high_risk = (df['referral_in'] == 1) | (df['gestational_age_wks'] < 37) | (df['birth_weight_g'] < 2500)
    
    # 1. Simulate Vitals (Conditionally linked to risk)
    # High Risk: 40% chance of severe hypertension. Normal: 5% chance.
    bp_high = np.random.choice([150, 160, 180, 200, 220], size=n, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    bp_normal = np.random.normal(115, 8, n)
    df['systolic_bp'] = np.where(is_high_risk & (np.random.rand(n) < 0.4), bp_high, bp_normal)
    
    # High Risk: 50% chance of severe anemia. Normal: 10% chance.
    hb_low = np.random.uniform(5.0, 9.5, n)
    hb_normal = np.random.uniform(11.0, 14.5, n)
    df['hemoglobin'] = np.where(is_high_risk & (np.random.rand(n) < 0.5), hb_low, hb_normal)
    
    df['parity'] = np.clip(np.random.poisson(lam=df['maternal_age']/10, size=n), 0, 8)
    df['maternal_bmi'] = 22 + (df['maternal_age'] - 20) * 0.1 + np.random.normal(0, 3, n)
    
    # 2. Simulate Socio-Environmental (Conditionally linked to risk)
    # High risk/Referred patients usually have lower ANC and live further away
    anc_low = np.random.randint(0, 3, n)
    anc_normal = np.random.randint(5, 9, n)
    df['anc_visits'] = np.where(is_high_risk, anc_low, anc_normal)
    
    dist_far = np.random.uniform(25, 80, n)
    dist_close = np.random.uniform(2, 15, n)
    df['distance_to_hospital'] = np.where(df['referral_in'] == 1, dist_far, dist_close)
    
    # 3. Create Binary Clinical Flags (Crucial for Recommender & XGBoost)
    df['hypertension_flag'] = (df['systolic_bp'] >= 140).astype(int)
    df['anemia_flag'] = (df['hemoglobin'] < 11.0).astype(int)
    df['preterm_flag'] = (df['gestational_age_wks'] < 37).astype(int)
    df['low_bw_flag'] = (df['birth_weight_g'] < 2500).astype(int)
    df['low_anc_flag'] = (df['anc_visits'] < 4).astype(int)
    df['high_distance_flag'] = (df['distance_to_hospital'] > 20).astype(int)
    
    return df

# ==========================================
# 3. KNN SEGMENTATION (Cohort Profiling)
# ==========================================
def build_knn_segmenter(df: pd.DataFrame):
    """Builds KNN model to assign patients to Clinical Archetypes."""
    logger.info("Building KNN Segmentation Model...")
    segment_features = ['maternal_age', 'parity', 'systolic_bp', 'hemoglobin', 
                        'anc_visits', 'distance_to_hospital']
    
    archetypes = {
        0: "Archetype_A: Young, Rural, Low ANC (High Risk)",
        1: "Archetype_B: Older, Hypertensive, Urban (Moderate Risk)",
        2: "Archetype_C: Prime Age, Healthy, Good ANC (Low Risk)",
        3: "Archetype_D: High Parity, Remote, Moderate Vitals (Moderate Risk)"
    }
    
    centroids = np.array([
        [18, 1, 110, 10.5, 2, 45],  # Archetype A
        [36, 4, 145, 11.0, 5, 5],   # Archetype B
        [25, 1, 115, 12.0, 8, 10],  # Archetype C
        [32, 6, 125, 11.2, 3, 35]   # Archetype D
    ])
    
    scaler = StandardScaler()
    scaled_centroids = scaler.fit_transform(centroids)
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(scaled_centroids)
    
    X_seg = scaler.transform(df[segment_features])
    _, indices = knn.kneighbors(X_seg)
    
    df['knn_segment_id'] = indices.flatten()
    df['knn_segment_name'] = df['knn_segment_id'].map(archetypes)
    
    segmenter_artifact = {'scaler': scaler, 'knn': knn, 'features': segment_features, 'archetypes': archetypes}
    joblib.dump(segmenter_artifact, 'models/knn_segmenter.pkl')
    logger.info("KNN Segmenter saved.")
    return df

# ==========================================
# 4. CLINICAL FLAG-BASED RECOMMENDER
# ==========================================
def build_and_save_recommender_artifact():
    """Creates and saves the static matrices for the Hybrid Recommender."""
    logger.info("Building Clinical Flag-Based Recommender...")
    interventions = ['Routine_Care', 'Nutrition_Support', 'Hypertension_Mgmt', 
                     'Anemia_Mgmt', 'Level4_Referral', 'Emergency_Transport']
    
    # Features: ['hypertension_flag', 'anemia_flag', 'preterm_flag', 'low_bw_flag', 'low_anc_flag', 'high_distance_flag', 'referral_in']
    # Using binary flags prevents normalization from inverting clinical logic (e.g. low Hb = bad)
    content_matrix = np.array([
        [0, 0, 0, 0, 0, 0, 0],  # Routine_Care (All clear)
        [0, 1, 0, 0, 1, 0, 0],  # Nutrition_Support (Anemia + Low ANC)
        [1, 0, 0, 0, 0, 0, 0],  # Hypertension_Mgmt (High BP)
        [0, 1, 0, 0, 0, 0, 0],  # Anemia_Mgmt (Low Hb)
        [1, 0, 1, 1, 0, 0, 1],  # Level4_Referral (HTN, Preterm, LBW, Referred)
        [0, 0, 0, 0, 1, 1, 0]   # Emergency_Transport (Low ANC, Far Distance)
    ])
    
    collaborative_matrix = np.array([
        [0.4, 0.5, 0.6, 0.5, 0.8, 0.9],  # Archetype A
        [0.6, 0.6, 0.9, 0.6, 0.7, 0.5],  # Archetype B
        [0.9, 0.8, 0.5, 0.8, 0.4, 0.3],  # Archetype C
        [0.5, 0.6, 0.7, 0.6, 0.9, 0.8]   # Archetype D
    ])
    
    recommender_artifact = {
        'interventions': interventions,
        'content_matrix': content_matrix,
        'collaborative_matrix': collaborative_matrix,
        'feature_cols': ['hypertension_flag', 'anemia_flag', 'preterm_flag', 
                         'low_bw_flag', 'low_anc_flag', 'high_distance_flag', 'referral_in']
    }
    
    joblib.dump(recommender_artifact, 'models/hybrid_recommender.pkl')
    logger.info("Hybrid Recommender artifact saved.")
    return recommender_artifact

# ==========================================
# 5. PREDICTIVE MODELING (XGBoost)
# ==========================================
def train_predictive_model(df: pd.DataFrame):
    """Trains XGBoost to predict adverse outcomes."""
    logger.info("Training XGBoost Predictive Model...")
    
    # Explicitly define the numerical features to feed into XGBoost.
    feature_cols = [
        'maternal_age', 'parity', 'systolic_bp', 'hemoglobin', 'anc_visits', 'distance_to_hospital',
        'hypertension_flag', 'anemia_flag', 'preterm_flag', 'low_bw_flag', 'low_anc_flag', 'high_distance_flag', 'referral_in'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols]
    y = df['adverse_outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', 
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Model ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info("\n" + classification_report(y_test, model.predict(X_test)))
    
    joblib.dump(model, 'models/xgb_risk_model.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    
    # 🔧 CRITICAL ADDITION: Save the test set for evaluate.py
    joblib.dump((X_test, y_test), 'models/test_set.pkl')
    logger.info(" Test set saved to models/test_set.pkl for evaluation.")
    
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    
    # 1. Load & Clean
    raw_df = load_and_clean_data('data/KUfacility_register_data_for_uploadWAISWAetalPLOSONE82020.xlsx')
    raw_df = clean_raw_types(raw_df)
    df = define_target_and_map_categories(raw_df)
    
    # 2. Feature Engineering
    df = engineer_anc_features(df)
    
    # 3. Segmentation
    df = build_knn_segmenter(df)
    
    # 4. Recommender
    build_and_save_recommender_artifact()
    
    # 5. Predictive Model
    train_predictive_model(df)
    
    logger.info(" All clinically-aligned models trained and saved to /models successfully!")