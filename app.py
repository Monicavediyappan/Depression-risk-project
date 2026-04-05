import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Depression Risk - AIDS Patients",
    page_icon="brain",
    layout="wide"
)

st.title("Depression Risk Prediction in AIDS Patients")
st.markdown(
    "**Quantum-Enhanced Stacking Ensemble | "
    "SHAP + LIME + DiCE | AUC: 0.98**"
)
st.markdown("---")

feature_names = [
    'CD4_Count','Viral_Load_log','ART_Adherence',
    'HIV_Duration_yrs','Opportunistic_Inf',
    'PHQ9_Score','GAD7_Score','Stress_Level',
    'Sleep_Quality','Age','Gender','Social_Support',
    'Stigma_Score','Employment_Status','Education_Level',
    'Income_Level','Substance_Use','Therapy_History',
    'Physical_Activity','Nutrition_Score'
]

@st.cache_resource
def load_models():
    with open("models/stack_model.pkl","rb")   as f: sm = pickle.load(f)
    with open("models/stack_quantum.pkl","rb") as f: sq = pickle.load(f)
    with open("models/scaler.pkl","rb")        as f: sc = pickle.load(f)
    with open("models/quantum_top.pkl","rb")   as f: qt = pickle.load(f)
    return sm, sq, sc, qt

try:
    stack_model, stack_quantum, scaler, quantum_top = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Models not loaded: {e}")

# Sidebar
st.sidebar.header("Enter Patient Details")

st.sidebar.subheader("HIV Clinical")
CD4_Count         = st.sidebar.slider("CD4 Count (cells/mm3)", 50,1200,500)
Viral_Load_log    = st.sidebar.slider("Viral Load log10",1.0,6.0,3.0,step=0.1)
ART_Adherence     = st.sidebar.selectbox("ART Adherence",[0,1],
    format_func=lambda x:"Non-Adherent" if x==0 else "Adherent")
HIV_Duration_yrs  = st.sidebar.slider("HIV Duration (years)",1,25,5)
Opportunistic_Inf = st.sidebar.selectbox("Opportunistic Infections",[0,1],
    format_func=lambda x:"No" if x==0 else "Yes")

st.sidebar.subheader("Mental Health")
PHQ9_Score        = st.sidebar.slider("PHQ-9 Score (0-27)",0,27,10)
GAD7_Score        = st.sidebar.slider("GAD-7 Score (0-21)",0,21,8)
Stress_Level      = st.sidebar.slider("Stress Level (1-10)",1,10,5)
Sleep_Quality     = st.sidebar.slider("Sleep Quality (1-10)",1,10,5)
Therapy_History   = st.sidebar.selectbox("Therapy History",[0,1],
    format_func=lambda x:"No" if x==0 else "Yes")

st.sidebar.subheader("Social and Demographic")
Age               = st.sidebar.slider("Age",18,70,35)
Gender            = st.sidebar.selectbox("Gender",[0,1],
    format_func=lambda x:"Female" if x==0 else "Male")
Social_Support    = st.sidebar.slider("Social Support (1-10)",1,10,5)
Stigma_Score      = st.sidebar.slider("Stigma Score (1-10)",1,10,5)
Employment_Status = st.sidebar.selectbox("Employment",[0,1],
    format_func=lambda x:"Unemployed" if x==0 else "Employed")
Education_Level   = st.sidebar.selectbox("Education",[1,2,3,4],
    format_func=lambda x:{1:"None",2:"Primary",
                          3:"Secondary",4:"Degree"}[x])
Income_Level      = st.sidebar.selectbox("Income",[1,2,3],
    format_func=lambda x:{1:"Low",2:"Medium",3:"High"}[x])

st.sidebar.subheader("Behavioral")
Substance_Use     = st.sidebar.selectbox("Substance Use",[0,1],
    format_func=lambda x:"No" if x==0 else "Yes")
Physical_Activity = st.sidebar.slider("Physical Activity (1-10)",1,10,5)
Nutrition_Score   = st.sidebar.slider("Nutrition Score (1-10)",1,10,5)

# Build input
input_data = pd.DataFrame([[
    CD4_Count, Viral_Load_log, ART_Adherence,
    HIV_Duration_yrs, Opportunistic_Inf,
    PHQ9_Score, GAD7_Score, Stress_Level,
    Sleep_Quality, Age, Gender, Social_Support,
    Stigma_Score, Employment_Status, Education_Level,
    Income_Level, Substance_Use, Therapy_History,
    Physical_Activity, Nutrition_Score
]], columns=feature_names)

if models_loaded:
    input_scaled = pd.DataFrame(
        scaler.transform(input_data), columns=feature_names
    )
    pred_full  = stack_model.predict(input_scaled)[0]
    prob_full  = stack_model.predict_proba(input_scaled)[0][1]
    pred_q     = stack_quantum.predict(input_scaled[quantum_top])[0]
    prob_q     = stack_quantum.predict_proba(
                     input_scaled[quantum_top]
                 )[0][1]

    # Results
    st.header("Prediction Result")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Full Model",
                  "HIGH RISK" if pred_full==1 else "LOW RISK")
    with c2:
        st.metric("Risk Probability", f"{prob_full:.1%}")
    with c3:
        st.metric("Quantum Model",
                  "HIGH RISK" if pred_q==1 else "LOW RISK")
    with c4:
        st.metric("Quantum Probability", f"{prob_q:.1%}")

    if pred_full == 1:
        st.error("HIGH RISK detected — clinical intervention recommended")
    else:
        st.success("LOW RISK — continue routine monitoring")

    # Risk gauge
    st.subheader("Risk Gauge")
    fig_g, ax_g = plt.subplots(figsize=(10,2))
    clr = "#e74c3c" if prob_full > 0.5 else "#2ecc71"
    ax_g.barh(["Risk"],[prob_full],   color=clr,     height=0.5)
    ax_g.barh(["Risk"],[1-prob_full], color="#ecf0f1",
              height=0.5, left=[prob_full])
    ax_g.axvline(0.5,color='orange',linewidth=2,linestyle='--')
    ax_g.set_xlim(0,1)
    ax_g.set_xticks([0,0.25,0.5,0.75,1.0])
    ax_g.set_xticklabels(['0%','25%','50%','75%','100%'])
    ax_g.set_title(f"Depression Risk: {prob_full:.1%}",
                   fontweight='bold')
    st.pyplot(fig_g)
    plt.close()

    st.markdown("---")

    # SHAP
    st.subheader("SHAP Explanation")
    try:
        rf_base   = stack_model.named_estimators_['rf']
        explainer = shap.TreeExplainer(rf_base)
        shap_vals = explainer.shap_values(input_scaled)
        if isinstance(shap_vals, list):
            shap_p = shap_vals[1][0]
        else:
            shap_p = np.array(shap_vals).flatten()
        if shap_p.ndim > 1:
            shap_p = shap_p[:,1]

        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP': shap_p
        }).sort_values('SHAP', key=abs, ascending=False).head(12)

        fig_s, ax_s = plt.subplots(figsize=(9,6))
        clrs_s = ['#e74c3c' if v > 0 else '#2ecc71'
                  for v in shap_df['SHAP']]
        ax_s.barh(shap_df['Feature'][::-1],
                  shap_df['SHAP'][::-1],
                  color=clrs_s[::-1], edgecolor='white')
        ax_s.axvline(0, color='black', linewidth=0.8)
        ax_s.set_title("Top 12 Features — SHAP Values\n"
                       "Red = increases risk  |  "
                       "Green = decreases risk",
                       fontweight='bold')
        ax_s.set_xlabel("SHAP Value")
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP unavailable: {e}")

    st.markdown("---")

    # Clinical report
    st.subheader("Clinical Report")
    risk_lbl = "HIGH RISK" if pred_full == 1 else "LOW RISK"
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(pd.DataFrame({
            "Item":   ["Risk Level","Probability",
                       "Quantum Model","AUC Score",
                       "Model Used"],
            "Result": [risk_lbl, f"{prob_full:.1%}",
                       "HIGH" if pred_q==1 else "LOW",
                       "0.98",
                       "Quantum Stacking Ensemble"]
        }), use_container_width=True)

    with col2:
        st.subheader("Recommendations")
        if pred_full==1 and PHQ9_Score>=15:
            st.error("URGENT: Immediate psychiatric referral")
        if pred_full==1 and PHQ9_Score>=10:
            st.error("Begin antidepressant evaluation")
        if ART_Adherence==0:
            st.error("URGENT: ART adherence counseling")
        if CD4_Count<200:
            st.error("URGENT: Immunology review needed")
        if Social_Support<4:
            st.info("Connect with support groups")
        if Sleep_Quality<4:
            st.info("Sleep disorder assessment recommended")
        if not (pred_full==1 or ART_Adherence==0
                or CD4_Count<200):
            st.success("Routine monitoring — no urgent action")

else:
    st.warning(
        "Models not found. Upload pkl files to models/ folder."
    )

st.markdown("---")
st.markdown(
    "**Model:** Quantum Stacking Ensemble | "
    "**AUC:** 0.98 | **XAI:** SHAP+LIME+DiCE | "
    "**Quantum:** PennyLane VQC + Qiskit QSVC | "
    "[GitHub]"
    "(https://github.com/Monicavediyappan/"
    "Depression-risk-project)"
)
