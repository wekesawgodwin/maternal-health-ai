# src/preprocessing.py
"""
Canonical data loading, cleaning, target definition, and feature engineering.

NOTE ON CONSOLIDATION: This file previously diverged from the feature-engineering
logic inlined in train.py - two different pipelines existed side by side
(different formulas, different filters, no shared source of truth). That's a
correctness risk: whichever one you happened to call would silently produce a
different dataset. This version is now the ONLY preprocessing pipeline; train.py
imports it directly instead of redefining its own copy.

Two prior bugs fixed here:
1. `define_target` compared `c_mother_status == 'Died'` case-sensitively, which
   would miss rows recorded as 'died' or 'DIED'. Now normalized to lowercase
   before comparison.
2. The old `engineer_anc_features` here (facility-level based) had much weaker
   signal than the risk-conditioned version, and `load_and_clean_data` didn't
   include 'BBA' (born-before-arrival) records and never coerced the
   object-typed numeric columns (referral_in, apgar_1, etc.), which crashes
   XGBoost with a ValueError on mixed dtypes. Both are fixed below.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Canonical feature list used by the predictive model (train.py, predict.py, evaluate.py
# all import this instead of re-typing the column list and risking drift).
MODEL_FEATURE_COLS: List[str] = [
    'maternal_age', 'parity', 'systolic_bp', 'hemoglobin', 'anc_visits', 'distance_to_hospital',
    'hypertension_flag', 'anemia_flag', 'preterm_flag', 'low_bw_flag', 'low_anc_flag',
    'high_distance_flag', 'referral_in'
]

# Continuous features PCA is run on (subset of the above, plus maternal_bmi which
# is not fed to the classifier directly but is informative for PCA/segmentation).
PCA_INPUT_COLS: List[str] = [
    'maternal_age', 'parity', 'systolic_bp', 'hemoglobin', 'maternal_bmi',
    'anc_visits', 'distance_to_hospital'
]

# Binary clinical flags consumed by the recommender.
RECOMMENDER_FLAG_COLS: List[str] = [
    'hypertension_flag', 'anemia_flag', 'preterm_flag', 'low_bw_flag',
    'low_anc_flag', 'high_distance_flag', 'referral_in'
]

NUMERIC_COERCE_COLS = [
    'referral_in', 'apgar_1', 'apgar_5', 'multiple', 'bba',
    'doc_abortion', 'doc_iufd', 'c_birth_weight_g2'
]

ADVERSE_BABY_STATUSES = [
    'Fresh_Still_Birth', 'Macerated_Still_Birth', 'Immediate_Neonatal_Death', 'Unknown_Still_Birth'
]

AGE_MAP = {'≤19': 17, '20-24': 22, '25-29': 27, '30-34': 32, '≥35': 38, 'missing': 25}
GA_MAP = {
    '<24wks': 20, '24-27wks': 25, '24-28 wks': 26, '28-30wks': 29, '28-30 weeks': 29,
    '31-33wks': 32, '34-36wks': 35, '37wks+': 39, '37+wks': 39
}
BW_MAP = {
    '<500g': 400, '500-999g': 750, '1000-1499g': 1250, '1500-1999g': 1750,
    '2000-2499g': 2250, '2500-2999g': 2750, '3000-3499g': 3250, '3500g+': 3750
}


# ==========================================
# 1. DATA LOADING & CLINICAL CLEANING
# ==========================================
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads raw maternity register data and applies clinical cleaning rules."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)

    # Filter for Kenya (country == 2) and actual Birth records (including
    # born-before-arrival, which are still live-birth-adjacent clinical events).
    df = df[(df['country'] == 2) & (df['record_type'].isin(['Birth', 'BBA']))].copy()

    df.drop_duplicates(subset=['id'], inplace=True)
    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def clean_raw_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw object/string columns with empty values into proper numeric
    types. Without this, XGBoost raises a ValueError on mixed dtypes the
    first time a column like `referral_in` contains a blank cell.
    """
    for col in NUMERIC_COERCE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df


def define_target_and_map_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Defines the adverse-outcome target and maps categorical variables to numeric midpoints."""
    # Normalize case before comparing - raw data has mixed 'Died'/'died' values.
    df['c_mother_status_clean'] = df['c_mother_status'].astype(str).str.lower()

    df['adverse_outcome'] = (
        (df['c_mother_status_clean'] == 'died') |
        (df['c_baby_status'].isin(ADVERSE_BABY_STATUSES))
    ).astype(int)

    df['maternal_age'] = df['mothers_age_cat'].map(AGE_MAP).fillna(25)
    df['gestational_age_wks'] = df['c_cat_ga'].map(GA_MAP).fillna(39)
    df['birth_weight_g'] = df['c_cat_bw'].map(BW_MAP).fillna(3250)
    df['apgar_1'] = df['apgar_1'].fillna(0)

    return df


# ==========================================
# 2. DOMAIN-DRIVEN FEATURE ENGINEERING
# ==========================================
def engineer_anc_features(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Simulates ANC features CONDITIONALLY on real clinical risk factors
    (referral_in, preterm, low birth weight) so the model learns correct
    clinical relationships instead of noise. In production this would be
    replaced by real ANC-register values (KHIS).
    """
    np.random.seed(random_state)
    n = len(df)

    is_high_risk = (df['referral_in'] == 1) | (df['gestational_age_wks'] < 37) | (df['birth_weight_g'] < 2500)

    bp_high = np.random.choice([150, 160, 180, 200, 220], size=n, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    bp_normal = np.random.normal(115, 8, n)
    df['systolic_bp'] = np.where(is_high_risk & (np.random.rand(n) < 0.4), bp_high, bp_normal)

    hb_low = np.random.uniform(5.0, 9.5, n)
    hb_normal = np.random.uniform(11.0, 14.5, n)
    df['hemoglobin'] = np.where(is_high_risk & (np.random.rand(n) < 0.5), hb_low, hb_normal)

    df['parity'] = np.clip(np.random.poisson(lam=df['maternal_age'] / 10, size=n), 0, 8)
    df['maternal_bmi'] = 22 + (df['maternal_age'] - 20) * 0.1 + np.random.normal(0, 3, n)

    anc_low = np.random.randint(0, 3, n)
    anc_normal = np.random.randint(5, 9, n)
    df['anc_visits'] = np.where(is_high_risk, anc_low, anc_normal)

    dist_far = np.random.uniform(25, 80, n)
    dist_close = np.random.uniform(2, 15, n)
    df['distance_to_hospital'] = np.where(df['referral_in'] == 1, dist_far, dist_close)

    df['hypertension_flag'] = (df['systolic_bp'] >= 140).astype(int)
    df['anemia_flag'] = (df['hemoglobin'] < 11.0).astype(int)
    df['preterm_flag'] = (df['gestational_age_wks'] < 37).astype(int)
    df['low_bw_flag'] = (df['birth_weight_g'] < 2500).astype(int)
    df['low_anc_flag'] = (df['anc_visits'] < 4).astype(int)
    df['high_distance_flag'] = (df['distance_to_hospital'] > 20).astype(int)

    return df


def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    """Main preprocessing orchestration: load -> coerce dtypes -> target -> engineer features."""
    df = load_and_clean_data(filepath)
    df = clean_raw_types(df)
    df = define_target_and_map_categories(df)
    df = engineer_anc_features(df)
    return df


# ==========================================
# 3. FEATURE RELATIONSHIPS (used by train.py's EDA step)
# ==========================================
def compute_correlations(df: pd.DataFrame, target_col: str = 'adverse_outcome') -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Returns (full correlation matrix, target correlations sorted, list of
    (feat_a, feat_b, r) pairs with |r| > 0.7). Mirrors the notebook's
    Section 4B analysis so train.py can log/plot the same diagnostics.
    """
    corr_features = PCA_INPUT_COLS + RECOMMENDER_FLAG_COLS + [target_col]
    corr_features = list(dict.fromkeys(corr_features))  # de-dupe, preserve order
    corr_matrix = df[corr_features].corr()

    target_corr = corr_matrix[target_col].drop(target_col).sort_values()

    predictor_corr = corr_matrix.drop(index=target_col, columns=target_col)
    high_corr_pairs = []
    cols = predictor_corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = predictor_corr.iloc[i, j]
            if abs(val) > 0.7:
                high_corr_pairs.append((cols[i], cols[j], round(float(val), 3)))

    return corr_matrix, target_corr, high_corr_pairs


# ==========================================
# 4. DIMENSIONALITY REDUCTION (PCA)
# ==========================================
def fit_pca(df: pd.DataFrame, n_components: int = 4, random_state: int = 42):
    """
    Fits a StandardScaler + PCA on PCA_INPUT_COLS and returns
    (scaler, pca, transformed_df_with_pca_cols).
    The fitted scaler/pca are what train.py persists to pca_transformer.pkl
    so predict.py can apply the identical transform at inference time.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[PCA_INPUT_COLS])

    pca = PCA(n_components=n_components, random_state=random_state)
    components = pca.fit_transform(X_scaled)

    df = df.copy()
    for i in range(n_components):
        df[f'pca_{i + 1}'] = components[:, i]

    return scaler, pca, df


def apply_pca(df: pd.DataFrame, scaler, pca) -> pd.DataFrame:
    """Applies an already-fitted scaler/PCA to new data (inference time)."""
    X_scaled = scaler.transform(df[PCA_INPUT_COLS])
    components = pca.transform(X_scaled)
    df = df.copy()
    for i in range(components.shape[1]):
        df[f'pca_{i + 1}'] = components[:, i]
    return df
