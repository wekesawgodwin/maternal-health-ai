# src/predict.py
import joblib
import numpy as np
import pandas as pd

def calculate_risk_score(probability: float) -> dict:
    """Converts model probability into clinical risk categories and actions."""
    if probability < 0.15:
        return {
            "risk_category": "Low Risk",
            "color": "green",
            "action": "Continue routine ANC. Standard delivery plan."
        }
    elif probability < 0.45:
        return {
            "risk_category": "Moderate Risk",
            "color": "orange",
            "action": "Increase ANC frequency to every 2 weeks. Monitor BP and fetal growth. Prepare emergency transport plan."
        }
    else:
        return {
            "risk_category": "High Risk",
            "color": "red",
            "action": "IMMEDIATE REFERRAL to Level 4/5 Hospital. Admit for close monitoring. Ensure blood cross-match and IV access."
        }

def predict_patient_risk(patient_data: pd.DataFrame):
    """Loads model and predicts risk for a single patient."""
    model = joblib.load('models/best_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    
    X_prep = preprocessor.transform(patient_data)
    probability = model.predict_proba(X_prep)[0, 1]
    
    risk_info = calculate_risk_score(probability)
    risk_info['probability'] = probability
    
    return risk_info