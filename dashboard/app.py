# dashboard/app.py
"""
Maternal & Child Health Clinical Decision Support System - Streamlit Dashboard.

BUG FIX: the previous version crashed with
    TypeError: 'MaternalSegmenter' object is not subscriptable
because it did `segmenter['scaler']` / `segmenter['knn']`, treating the saved
artifact as a plain dict. `models/knn_segmenter.pkl` now stores a fitted
`MaternalSegmenter` *object* (see src/segment.py), not a dict - so this
dashboard now calls `segmenter.predict_segments(df)` instead of manually
reaching into internals that may not even exist as dict keys.

DESIGN CHANGE: rather than re-deriving flag calculations, threshold logic,
and recommendation scoring a third time in the UI layer (the original app.py,
train.py, and predict.py had all quietly drifted from each other this way),
this dashboard now calls `MaternalRiskPredictor` from src/predict.py for the
entire risk + segment + recommendation pipeline. One code path, used
everywhere - the UI can't drift out of sync with the backend again.
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. PATH RESOLUTION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'

sys.path.append(str(PROJECT_ROOT / 'src'))
from predict import MaternalRiskPredictor          # noqa: E402
from recommend import risk_tier as compute_risk_tier  # noqa: E402
from preprocessing import AGE_MAP, GA_MAP, BW_MAP    # noqa: E402
from utils import load_model_metadata                # noqa: E402

# ==========================================
# 2. PAGE CONFIG & THEME
# ==========================================
st.set_page_config(page_title="Maternal CDSS - Kenya", page_icon="🤰", layout="wide",
                    initial_sidebar_state="expanded")

PALETTE = {
    "bg": "#F6F7FB",
    "card": "#FFFFFF",
    "ink": "#1B1F3B",
    "muted": "#6B7280",
    "primary": "#6C5CE7",
    "primary_dark": "#4834D4",
    "accent": "#00B8A9",
    "low": "#2ECC71",
    "elevated": "#F5C518",
    "high": "#FF8C42",
    "critical": "#E64848",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background: linear-gradient(180deg, #F6F7FB 0%, #EEF0FA 100%);
}}

/* ---------- Hero header ---------- */
.hero {{
    background: linear-gradient(120deg, {PALETTE['primary_dark']} 0%, {PALETTE['primary']} 55%, {PALETTE['accent']} 130%);
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 30px rgba(76, 52, 212, 0.25);
}}
.hero h1 {{
    font-size: 2.0rem;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}}
.hero p {{
    font-size: 1.0rem;
    opacity: 0.92;
    margin: 0;
    font-weight: 400;
}}
.hero .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.8rem;
    letter-spacing: 0.02em;
}}

/* ---------- Generic card ---------- */
.card {{
    background: {PALETTE['card']};
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 4px 18px rgba(27, 31, 59, 0.06);
    border: 1px solid rgba(27, 31, 59, 0.04);
    margin-bottom: 1rem;
    height: 100%;
}}
.card h4 {{
    margin-top: 0;
    color: {PALETTE['ink']};
    font-weight: 700;
    font-size: 1.0rem;
}}

/* ---------- Risk tier badges ---------- */
.tier-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    font-weight: 800;
    font-size: 1.15rem;
    color: white;
}}
.tier-LOW {{ background: {PALETTE['low']}; }}
.tier-ELEVATED {{ background: {PALETTE['elevated']}; color: #4A3B00; }}
.tier-HIGH {{ background: {PALETTE['high']}; }}
.tier-CRITICAL {{ background: {PALETTE['critical']}; }}

/* ---------- Flag chips ---------- */
.chip {{
    display: inline-block;
    background: #FDEDEC;
    color: #C0392B;
    border: 1px solid #F5C6C2;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.15rem 0.3rem 0.15rem 0;
}}
.chip-clear {{
    background: #EAFAF1;
    color: #1E8449;
    border: 1px solid #C9F0DA;
}}

/* ---------- Intervention card ---------- */
.intervention-card {{
    background: {PALETTE['card']};
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    border-left: 5px solid {PALETTE['primary']};
    box-shadow: 0 3px 14px rgba(27, 31, 59, 0.06);
    margin-bottom: 0.9rem;
}}
.intervention-card h4 {{
    margin: 0 0 0.3rem 0;
    color: {PALETTE['ink']};
}}
.intervention-meta {{
    color: {PALETTE['muted']};
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}}

/* ---------- Checklist ---------- */
.check-pass {{ color: #1E8449; font-weight: 700; }}
.check-fail {{ color: #C0392B; font-weight: 700; }}

/* Section headers */
.section-title {{
    font-weight: 800;
    font-size: 1.15rem;
    color: {PALETTE['ink']};
    margin: 1.2rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD BACKEND (cached)
# ==========================================
@st.cache_resource(show_spinner="Loading trained models...")
def load_predictor() -> MaternalRiskPredictor:
    try:
        return MaternalRiskPredictor(models_dir=MODELS_DIR).load()
    except FileNotFoundError as e:
        st.error(f"❌ Model artifacts not found: {e}\n\nPlease run `python src/train.py` from the project root first.")
        st.stop()


predictor = load_predictor()

INTERVENTION_ICONS = {
    "Routine Care": "🩺", "Nutrition Support": "🥗", "Hypertension Mgmt": "💊",
    "Anemia Mgmt": "🩸", "Level4 Referral": "🏥", "Emergency Transport": "🚑",
}
TIER_ICONS = {"LOW": "🟢", "ELEVATED": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}

# ==========================================
# 4. HERO HEADER
# ==========================================
st.markdown(f"""
<div class="hero">
    <h1>🤰 Maternal &amp; Child Health Clinical Decision Support System</h1>
    <p>Domain-aligned triage, hybrid recommender &amp; model evaluation dashboard — KHIS/DHIS2 prototype</p>
    <span class="badge">Deployed model: {predictor.metadata.get('model_name', '—')} &nbsp;|&nbsp;
    Decision threshold: {predictor.threshold:.3f} &nbsp;|&nbsp;
    Recall target: {(predictor.metadata.get('min_recall_target') or 0):.0%}</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🩺  Clinical Triage & Recommender", "📊  Model Evaluation & Deployment"])

# ==========================================
# TAB 1: CLINICAL TRIAGE & RECOMMENDER
# ==========================================
with tab1:
    st.sidebar.markdown("### 📋 Patient Intake & ANC Details")

    with st.sidebar.form("patient_intake"):
        st.markdown("**Demographics & History**")
        age_cat = st.selectbox("Maternal Age Category", list(AGE_MAP.keys()), index=1)
        parity = st.number_input("Parity (Previous Births)", 0, 15, 1)
        referral_in = st.selectbox("Referred from another facility?", ["No", "Yes"])

        st.markdown("**Clinical Vitals (Intrapartum / ANC)**")
        ga_options = [k for k in GA_MAP.keys() if 'weeks' not in k]  # drop duplicate-labeled variants
        gestational_age = st.selectbox("Gestational Age", ga_options, index=len(ga_options) - 1)
        bw_options = list(BW_MAP.keys())
        birth_weight = st.selectbox("Birth Weight Category", bw_options, index=len(bw_options) - 1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 220, 115)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 16.0, 11.5, step=0.1)

        st.markdown("**Socio-Environmental**")
        anc_visits = st.number_input("ANC Visits Completed", 0, 15, 6)
        distance = st.number_input("Distance to Referral Hospital (km)", 0.0, 100.0, 10.0)

        with st.expander("Advanced (optional)"):
            maternal_bmi = st.number_input("Maternal BMI", 12.0, 50.0, 22.0, step=0.1)

        submitted = st.form_submit_button("🩺  Analyze Patient Risk & Pathway", use_container_width=True)

    if not submitted:
        st.info("👈 Fill in the patient intake form in the sidebar and click **Analyze Patient Risk & Pathway** "
                "to generate a full clinical decision-support report.")
    else:
        patient_input = {
            'maternal_age': AGE_MAP[age_cat],
            'parity': parity,
            'systolic_bp': systolic_bp,
            'hemoglobin': hemoglobin,
            'anc_visits': anc_visits,
            'distance_to_hospital': distance,
            'referral_in': 1 if referral_in == "Yes" else 0,
            'gestational_age_wks': GA_MAP[gestational_age],
            'birth_weight_g': BW_MAP[birth_weight],
            'maternal_bmi': maternal_bmi,
        }

        with st.spinner("Running risk model, segmentation, and recommender..."):
            report = predictor.predict(patient_input)

        prob = report['risk_probability']
        tier = report['risk_tier']
        threshold = report['decision_threshold']

        st.markdown('<div class="section-title">📊 Risk Assessment</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.1, 1, 1.2])

        with col1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <h4>Risk Tier</h4>
                <div class="tier-badge tier-{tier}">{TIER_ICONS.get(tier, '')} &nbsp;{tier}</div>
                <p style="color:{PALETTE['muted']}; margin-top:0.8rem; font-size:0.88rem;">
                    Predicted probability of an adverse maternal/perinatal outcome:
                    <b>{prob*100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            action_map = {
                "CRITICAL": ("🚨", "Immediate Medical Officer review. Prepare for emergency referral & transport."),
                "HIGH": ("⚠️", "Escalate for clinician review today. Initiate flagged interventions below."),
                "ELEVATED": ("⚡", "Increase monitoring frequency. Review recommended interventions."),
                "LOW": ("✅", "Continue routine antenatal/postnatal care schedule."),
            }
            icon, msg = action_map.get(tier, ("ℹ️", ""))
            st.markdown(f"**{icon} Recommended Action:** {msg}")

        with col2:
            gauge_zones = [
                {"range": [0, threshold * 0.5], "color": PALETTE["low"]},
                {"range": [threshold * 0.5, threshold], "color": PALETTE["elevated"]},
                {"range": [threshold, max(threshold * 2, 0.6)], "color": PALETTE["high"]},
                {"range": [max(threshold * 2, 0.6), 1.0], "color": PALETTE["critical"]},
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 34}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': PALETTE['ink'], 'thickness': 0.25},
                    'steps': [{'range': [z['range'][0] * 100, z['range'][1] * 100], 'color': z['color']} for z in gauge_zones],
                    'threshold': {'line': {'color': PALETTE['ink'], 'width': 3},
                                  'thickness': 0.8, 'value': threshold * 100},
                },
            ))
            fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=10),
                               paper_bgcolor='rgba(0,0,0,0)', font={'family': 'Inter'})
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Black line marks the deployed decision threshold ({threshold:.2f}), "
                       f"tuned so the model catches ≥{(predictor.metadata.get('min_recall_target') or 0):.0%} "
                       f"of true adverse outcomes.")

        with col3:
            st.markdown(f"""
            <div class="card">
                <h4>👥 Patient Archetype (KNN Cohort)</h4>
                <p style="font-weight:700; color:{PALETTE['primary_dark']}; font-size:1.02rem;">{report['archetype']}</p>
                <p style="color:{PALETTE['muted']}; font-size:0.85rem;">
                    Nearest clinical archetype based on Kenyan MoH risk profiling - used to weight
                    which interventions have historically worked for similar patients.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Active Clinical Flags**")
            if report['active_flags'] == ["No active clinical risk flags"]:
                st.markdown('<span class="chip chip-clear">✓ No active risk flags</span>', unsafe_allow_html=True)
            else:
                chips = "".join(f'<span class="chip">⚑ {f}</span>' for f in report['active_flags'])
                st.markdown(chips, unsafe_allow_html=True)

        st.markdown('<div class="section-title">💡 Hybrid Recommender: Recommended Interventions</div>',
                    unsafe_allow_html=True)
        st.caption("Ranked by a hybrid of **clinical rule-matching** (content-based) and **historical cohort "
                   "success** (collaborative filtering). Each includes a concrete action, timeline, and rationale.")

        rec_cols = st.columns(len(report['recommendations']))
        for i, rec in enumerate(report['recommendations']):
            icon = INTERVENTION_ICONS.get(rec['intervention'], "🔹")
            with rec_cols[i]:
                st.markdown(f"""
                <div class="intervention-card">
                    <h4>{icon} {i+1}. {rec['intervention']}</h4>
                    <div class="intervention-meta">Confidence: <b>{rec['confidence']}</b></div>
                    <p style="font-size:0.88rem;"><b>Action:</b> {rec['action']}</p>
                    <p style="font-size:0.88rem;"><b>Timeline:</b> {rec['timeline']}</p>
                    <p style="font-size:0.82rem; color:{PALETTE['muted']};"><i>{rec['rationale']}</i></p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("🔍 Score breakdown"):
                    st.write(f"**Clinical rule match (content):** {rec['content_contrib']*100:.1f}%")
                    st.write(f"**Cohort success rate (collaborative):** {rec['collab_contrib']*100:.1f}%")
                    st.write(f"**Hybrid score:** {rec['hybrid_score']*100:.1f}%")

# ==========================================
# TAB 2: MODEL EVALUATION & DEPLOYMENT
# ==========================================
with tab2:
    st.markdown('<div class="section-title">📊 Model Evaluation & Deployment Readiness</div>', unsafe_allow_html=True)
    st.caption("Clinical safety metrics evaluated on the hold-out test set, at the model's tuned decision threshold "
               "(not the sklearn default of 0.5).")

    metrics_path = MODELS_DIR / 'evaluation_metrics.json'

    if not metrics_path.exists():
        st.warning("⚠️ Evaluation metrics not found. Please run `python src/evaluate.py` from the project root first.")
    else:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        target = metrics.get('min_recall_target') or 0.90

        st.markdown("#### 1. Core Clinical Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.1f}%",
                  help=f"Out of all actual adverse outcomes, how many did we catch? (Deployment gate: ≥{target:.0%})")
        m2.metric("Precision", f"{metrics['precision']*100:.1f}%",
                  help="Out of all flagged high-risk patients, how many were truly high-risk. "
                       "Expected to be low at a high-recall operating point - see note below.")
        m3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="Overall discrimination ability, threshold-independent.")
        m4.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", help="Ranking quality on the rare minority class.")

        st.markdown("#### 2. Clinical Impact Breakdown")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives (Caught)", metrics['true_positives'])
        c2.metric("False Negatives (Missed)", metrics['false_negatives'])
        c3.metric("False Positives (Extra Checks)", metrics['false_positives'])
        c4.metric("True Negatives (Cleared)", metrics['true_negatives'])

        st.markdown("#### 3. Deployment Gate")
        st.caption("In maternal triage, **recall is the hard safety gate** - a missed adverse outcome is far "
                   "costlier than an unnecessary check-up. Precision/PR-AUC are shown as context, not gates: "
                   "they are *expected* to be modest at a ≥90% recall operating point on a ~7% base-rate outcome.")

        recall_pass = metrics['recall'] >= target
        if recall_pass:
            st.markdown(f'<p class="check-pass">✅ PASS — Recall {metrics["recall"]*100:.1f}% meets the '
                        f'{target:.0%} clinical safety target.</p>', unsafe_allow_html=True)
            st.success("🎉 **Deployment gate met.** Model is approved for DHIS2/KHIS pilot integration on this metric.")
        else:
            st.markdown(f'<p class="check-fail">❌ FAIL — Recall {metrics["recall"]*100:.1f}% is below the '
                        f'{target:.0%} target.</p>', unsafe_allow_html=True)
            st.warning("⚠️ Model requires further tuning (lower threshold / more data / new features) "
                       "before clinical deployment.")

        with st.expander("ℹ️ Why is precision so low if this model is 'good'?"):
            st.write(
                f"At this threshold ({metrics['decision_threshold']:.3f}), the model flags many patients who "
                f"will NOT go on to have an adverse outcome — precision is only {metrics['precision']*100:.1f}%. "
                "This is the accepted clinical trade-off for a triage-screening tool: a false alarm costs "
                "an extra ANC check-up (low risk, low cost); a missed adverse outcome does not have an "
                "equivalent recovery. The deployment gate is therefore recall, not precision."
            )

        st.markdown("#### 4. Model Comparison")
        comp_path = MODELS_DIR / 'model_comparison.json'
        if comp_path.exists():
            with open(comp_path) as f:
                comparison = json.load(f)
            comp_df = pd.DataFrame({
                name: {'Recall': v['recall'], 'Precision': v['precision'], 'Threshold': v['threshold']}
                for name, v in comparison.items()
            }).T
            comp_df.index.name = "Model"
            st.dataframe(
                comp_df.style.format({'Recall': '{:.1%}', 'Precision': '{:.1%}', 'Threshold': '{:.3f}'})
                .background_gradient(subset=['Recall'], cmap='Greens')
                .background_gradient(subset=['Precision'], cmap='Blues'),
                use_container_width=True,
            )
            st.caption(f"**{metrics['model_name']}** was selected: it clears the recall gate with the best "
                       "precision among all tuned candidates.")

        st.markdown("#### 5. Evaluation Visualizations")
        viz_col1, viz_col2 = st.columns(2)
        viz_files = [
            ("Confusion Matrix", 'eval_cm.png'), ("Precision-Recall Curve", 'eval_pr.png'),
            ("ROC Curve", 'eval_roc.png'), ("Calibration Curve (Trust Metric)", 'eval_calib.png'),
        ]
        for i, (label, fname) in enumerate(viz_files):
            target_col = viz_col1 if i % 2 == 0 else viz_col2
            with target_col:
                st.markdown(f"**{label}**")
                fpath = FIGURES_DIR / fname
                if fpath.exists():
                    st.image(str(fpath), use_container_width=True)
                else:
                    st.info("Plot not generated yet - run `python src/evaluate.py`.")

        extra_col1, extra_col2 = st.columns(2)
        for target_col, fname, label in [
            (extra_col1, 'feature_correlations.png', 'Feature Correlations'),
            (extra_col2, 'pca_analysis.png', 'PCA Analysis'),
        ]:
            with target_col:
                fpath = FIGURES_DIR / fname
                if fpath.exists():
                    st.markdown(f"**{label}**")
                    st.image(str(fpath), use_container_width=True)

st.markdown("---")
st.caption("Maternal & Child Health CDSS Prototype · Built on the Waiswa et al. (PLOS ONE, 2020) "
           "Kenyan facility register dataset · For pilot/research use, not a certified medical device.")
