# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path

# ==========================================
# 1. DYNAMIC PATH RESOLUTION
# ==========================================
# Ensures the app finds models/figures regardless of where it's executed from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'

# Add src to path to import our custom modules
sys.path.append(str(PROJECT_ROOT / 'src'))
from recommend import recommend_interventions

# ==========================================
# 2. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Maternal CDSS - Kenya", page_icon="🤰", layout="wide")

st.markdown("""
    <style>
    .risk-high { color: #d62728; font-weight: bold; font-size: 26px; }
    .risk-mod { color: #ff7f0e; font-weight: bold; font-size: 26px; }
    .risk-low { color: #2ca02c; font-weight: bold; font-size: 26px; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🤰 Maternal & Child Health Clinical Decision Support System")
st.markdown("**Domain-Aligned Triage, Recommender & Model Evaluation Dashboard (KHIS Prototype)**")
st.markdown("---")

# ==========================================
# 3. LOAD MODELS (Cached for performance)
# ==========================================
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load(MODELS_DIR / 'xgb_risk_model.pkl')
        segmenter = joblib.load(MODELS_DIR / 'knn_segmenter.pkl')
        feature_names = joblib.load(MODELS_DIR / 'feature_names.pkl')
        return xgb_model, segmenter, feature_names
    except FileNotFoundError:
        st.error("❌ Models not found. Please run `python src/train.py` from the root directory first.")
        st.stop()

xgb_model, segmenter, feature_names = load_models()

# ==========================================
# 4. CREATE TABBED INTERFACE
# ==========================================
tab1, tab2 = st.tabs(["🩺 Clinical Triage & Recommender", "📊 Model Evaluation & Deployment"])

# ==========================================
# TAB 1: CLINICAL TRIAGE & RECOMMENDER
# ==========================================
with tab1:
    st.sidebar.header("📋 Patient Intake & ANC Details")
    
    with st.sidebar.form("patient_intake"):
        st.subheader("Demographics & History")
        age_cat = st.selectbox("Maternal Age Category", ['≤19', '20-24', '25-29', '30-34', '≥35'])
        parity = st.number_input("Parity (Previous Births)", 0, 15, 1)
        referral_in = st.selectbox("Referred from another facility?", ["No", "Yes"])
        
        st.subheader("Clinical Vitals (Intrapartum / ANC)")
        # Mapped strictly to Data Dictionary categories
        gestational_age = st.selectbox("Gestational Age", ['<24wks', '24-27wks', '28-30wks', '31-33wks', '34-36wks', '37wks+'])
        birth_weight = st.selectbox("Birth Weight Category", ['<500g', '500-999g', '1000-1499g', '1500-1999g', '2000-2499g', '2500-2999g', '3000-3499g', '3500g+'])
        
        systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 220, 115)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 16.0, 11.5)
        
        st.subheader("Socio-Environmental")
        anc_visits = st.number_input("ANC Visits Completed", 0, 15, 6)
        distance = st.number_input("Distance to Referral Hospital (km)", 0.0, 100.0, 10.0)
        
        submitted = st.form_submit_button("🩺 Analyze Patient Risk & Pathway")

    if submitted:
        # --- A. Map Categories to Numeric ---
        age_map = {'≤19': 17, '20-24': 22, '25-29': 27, '30-34': 32, '≥35': 38}
        ga_map = {'<24wks': 20, '24-27wks': 25, '28-30wks': 29, '31-33wks': 32, '34-36wks': 35, '37wks+': 39}
        bw_map = {'<500g': 400, '500-999g': 750, '1000-1499g': 1250, '1500-1999g': 1750, 
                  '2000-2499g': 2250, '2500-2999g': 2750, '3000-3499g': 3250, '3500g+': 3750}
        
        ga_wks = ga_map[gestational_age]
        bw_g = bw_map[birth_weight]
        is_ref = 1 if referral_in == "Yes" else 0
        
        # --- B. Calculate Binary Clinical Flags (MoH Guidelines) ---
        htn_flag = 1 if systolic_bp >= 140 else 0
        anemia_flag = 1 if hemoglobin < 11.0 else 0
        preterm_flag = 1 if ga_wks < 37 else 0
        low_bw_flag = 1 if bw_g < 2500 else 0
        low_anc_flag = 1 if anc_visits < 4 else 0
        high_dist_flag = 1 if distance > 20 else 0
        
        # --- C. Construct DataFrames for different models ---
        # 1. For XGBoost (Needs 13 specific features)
        xgb_data = {
            'maternal_age': [age_map[age_cat]], 'parity': [parity], 
            'systolic_bp': [systolic_bp], 'hemoglobin': [hemoglobin],
            'anc_visits': [anc_visits], 'distance_to_hospital': [distance],
            'hypertension_flag': [htn_flag], 'anemia_flag': [anemia_flag],
            'preterm_flag': [preterm_flag], 'low_bw_flag': [low_bw_flag],
            'low_anc_flag': [low_anc_flag], 'high_distance_flag': [high_dist_flag],
            'referral_in': [is_ref]
        }
        df_xgb = pd.DataFrame(xgb_data)
        # Safety check to ensure exact schema match
        for col in feature_names:
            if col not in df_xgb.columns: df_xgb[col] = 0
        df_xgb = df_xgb[feature_names]
        
        # 2. For KNN Segmenter (Needs 6 continuous features)
        seg_data = {
            'maternal_age': [age_map[age_cat]], 'parity': [parity],
            'systolic_bp': [systolic_bp], 'hemoglobin': [hemoglobin],
            'anc_visits': [anc_visits], 'distance_to_hospital': [distance]
        }
        df_seg = pd.DataFrame(seg_data)
        
        # 3. For Recommender (Needs 7 binary flags)
        rec_data = {
            'hypertension_flag': [htn_flag], 'anemia_flag': [anemia_flag],
            'preterm_flag': [preterm_flag], 'low_bw_flag': [low_bw_flag],
            'low_anc_flag': [low_anc_flag], 'high_distance_flag': [high_dist_flag],
            'referral_in': [is_ref]
        }
        df_rec = pd.DataFrame(rec_data)
        
        # --- D. Run Inference Pipelines ---
        # 1. Run KNN Segmentation
        scaler = segmenter['scaler']
        knn = segmenter['knn']
        X_seg = scaler.transform(df_seg[segmenter['features']])
        _, indices = knn.kneighbors(X_seg)
        
        segment_id = int(indices.flatten()[0])
        segment_name = segmenter['archetypes'][segment_id]
        
        # Attach segment ID to Recommender DataFrame
        df_rec['knn_segment_id'] = segment_id
        
        # 2. Run XGBoost Prediction (Cast to native float to prevent Streamlit errors)
        risk_prob = float(xgb_model.predict_proba(df_xgb)[:, 1][0])
        
        # 3. Run Hybrid Recommender
        recommendations = recommend_interventions(df_rec, alpha=0.7)
        
        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Risk Stratification")
            if risk_prob > 0.45:
                st.markdown(f'<p class="risk-high">HIGH RISK ({risk_prob*100:.1f}%)</p>', unsafe_allow_html=True)
                st.error("⚠️ **Action:** Immediate Medical Officer Review. Prepare for emergency referral.")
            elif risk_prob > 0.15:
                st.markdown(f'<p class="risk-mod">MODERATE RISK ({risk_prob*100:.1f}%)</p>', unsafe_allow_html=True)
                st.warning("⚡ **Action:** Increase monitoring frequency. Review recommended interventions.")
            else:
                st.markdown(f'<p class="risk-low">LOW RISK ({risk_prob*100:.1f}%)</p>', unsafe_allow_html=True)
                st.success("✅ **Action:** Continue routine postnatal/antenatal care.")
                
        with col2:
            st.subheader("👥 Patient Archetype (KNN)")
            st.info(f"**{segment_name}**")
            st.caption("Contextualizes the patient's socio-demographic and clinical risk profile based on historical Kenyan cohorts.")
            
        with col3:
            st.subheader("🎯 Clinical Pathway")
            st.metric("Predicted Adverse Outcome Probability", f"{risk_prob*100:.1f}%")
            st.progress(min(risk_prob, 1.0))

        st.markdown("---")
        st.subheader("💡 Hybrid Recommender: Top 3 Clinical Interventions")
        st.caption("Recommendations are driven by **Binary Clinical Flags** (e.g., Anemia Flag = 1 if Hb < 11) ensuring strict adherence to MoH guidelines.")
        
        rec_cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            with rec_cols[i]:
                st.markdown(f"### {i+1}. {rec['intervention']}")
                # Cast to float to ensure Streamlit metric renders correctly
                st.metric("Hybrid Confidence Score", f"{float(rec['hybrid_score'])*100:.1f}%")
                
                with st.expander("🔍 Why this intervention?"):
                    st.write(f"**Clinical Match (Content):** {float(rec['content_contrib'])*100:.1f}%")
                    st.write(f"**Cohort Success (Collaborative):** {float(rec['collab_contrib'])*100:.1f}%")


# ==========================================
# TAB 2: MODEL EVALUATION & DEPLOYMENT
# ==========================================
with tab2:
    st.header("📊 Model Evaluation & Deployment Readiness")
    st.markdown("Clinical safety metrics evaluated on the hold-out test set. This ensures the AI is safe for real-world KHIS/DHIS2 integration.")
    
    metrics_path = MODELS_DIR / 'evaluation_metrics.json'
    
    if not metrics_path.exists():
        st.warning("⚠️ Evaluation metrics not found. Please run `python src/evaluate.py` from the root directory to generate them.")
    else:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # 1. Key Clinical Metrics
        st.subheader("1. Core Clinical Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.1f}%", help="Out of all actual adverse outcomes, how many did we catch? (Target > 85%)")
        m_col2.metric("Precision", f"{metrics['precision']*100:.1f}%", help="Out of all flagged high-risk, how many were truly high-risk?")
        m_col3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="Overall ability to distinguish between normal and adverse outcomes.")
        m_col4.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", help="Performance on the rare minority class (Target > 0.30).")
        
        # 2. Confusion Matrix Breakdown
        st.subheader("2. Clinical Impact Breakdown")
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        cm_col1.metric("True Positives (Caught)", metrics['true_positives'], delta="Lives Saved")
        cm_col2.metric("False Negatives (Missed)", metrics['false_negatives'], delta="Critical Misses", delta_color="inverse")
        cm_col3.metric("False Positives (Alerts)", metrics['false_positives'], delta="Extra Checks")
        cm_col4.metric("True Negatives (Cleared)", metrics['true_negatives'], delta="Routine Care")
        
        # 3. Deployment Readiness Checklist
        st.subheader("3. Deployment Readiness Checklist")
        fn_rate = metrics['false_negatives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 1.0
        
        checks = {
            "Recall (Sensitivity) >= 85%": metrics['recall'] >= 0.85,
            "Precision-Recall AUC >= 0.30": metrics['pr_auc'] >= 0.30,
            "False Negative Rate < 15%": fn_rate < 0.15
        }
        
        for check, passed in checks.items():
            if passed:
                st.success(f"✅ **PASS**: {check}")
            else:
                st.error(f"❌ **FAIL**: {check}")
                
        all_pass = all(checks.values())
        if all_pass:
            st.balloons()
            st.success("🎉 **CONCLUSION**: Model meets clinical safety thresholds. Approved for DHIS2/KHIS pilot integration.")
        else:
            st.warning("⚠️ **CONCLUSION**: Model requires further tuning before clinical deployment.")
            
        # 4. Visualizations
        st.subheader("4. Evaluation Visualizations")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("**Confusion Matrix**")
            if (FIGURES_DIR / 'eval_cm.png').exists():
                st.image(FIGURES_DIR / 'eval_cm.png', use_column_width=True)
            else:
                st.info("Plot not generated yet.")
                
            st.markdown("**Precision-Recall Curve**")
            if (FIGURES_DIR / 'eval_pr.png').exists():
                st.image(FIGURES_DIR / 'eval_pr.png', use_column_width=True)
            else:
                st.info("Plot not generated yet.")
                
        with viz_col2:
            st.markdown("**ROC Curve**")
            if (FIGURES_DIR / 'eval_roc.png').exists():
                st.image(FIGURES_DIR / 'eval_roc.png', use_column_width=True)
            else:
                st.info("Plot not generated yet.")
                
            st.markdown("**Calibration Curve (Trust Metric)**")
            if (FIGURES_DIR / 'eval_calib.png').exists():
                st.image(FIGURES_DIR / 'eval_calib.png', use_column_width=True)
            else:
                st.info("Plot not generated yet.")