import streamlit as st
import pandas as pd
import joblib
import numpy as np

#Load the Model and the Scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Helper to map real age to the dataset's 1-13 scale
def convert_age(age):
    if age < 25: return 1
    elif age < 30: return 2
    elif age < 35: return 3
    elif age < 40: return 4
    elif age < 45: return 5
    elif age < 50: return 6
    elif age < 55: return 7
    elif age < 60: return 8
    elif age < 65: return 9
    elif age < 70: return 10
    elif age < 75: return 11
    elif age < 80: return 12
    else: return 13

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Health Risk Predictor")
st.markdown("Enter your metrics to assess your risk based on lifestyle and health indicators.")

# Major indicators (Top 6 most impactful) ---
st.header("Core Health Metrics")
col1, col2 = st.columns(2)

with col1:
    user_age = st.number_input("How old are you?", min_value=18, max_value=110, value=30)
    bmi = st.number_input("Your BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    phys_hlth = st.slider("Days of poor physical health (Last 30 days)", 0, 30, 0)

with col2:
    high_bp = st.selectbox("Do you have High Blood Pressure?", ["No", "Yes"])
    high_chol = st.selectbox("Do you have High Cholesterol?", ["No", "Yes"])
    gen_hlth = st.slider("General Health Rating (1=Excellent, 5=Poor)", 1, 5, 2)

# Logical conversions for the model
age_cat = convert_age(user_age)
high_bp_num = 1 if high_bp == "Yes" else 0
high_chol_num = 1 if high_chol == "Yes" else 0

# advanced option (hidden to keep the UI clean)
with st.expander("Additional Lifestyle & History"):
    st.info("The following are set to common defaults. Adjust them for a more precise prediction.")
    col3, col4 = st.columns(2)
    
    with col3:
        diff_walk = st.checkbox("Difficulty walking or climbing stairs?", value=False)
        heart_dis = st.checkbox("History of heart disease or attack?", value=False)
        stroke = st.checkbox("History of stroke?", value=False)
        smoker = st.checkbox("Smoked 100+ cigarettes in lifetime?", value=False)
        phys_act = st.checkbox("Regular physical activity?", value=True)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    
    with col4:
        ment_hlth = st.slider("Days of poor mental health (Last 30 days)", 0, 30, 0)
        hvy_alc = st.checkbox("Heavy alcohol consumption?", value=False)
        healthcare = st.checkbox("Have health insurance?", value=True)
        chol_check = st.checkbox("Had a cholesterol check recently?", value=True)
        fruits = st.checkbox("Eat fruit daily?", value=True)
        veggies = st.checkbox("Eat vegetables daily?", value=True)
        doc_cost = st.checkbox("Skipped doctor visit due to cost?", value=False)
        edu = st.slider("Education Level (1-6)", 1, 6, 4)
        income = st.slider("Income Scale (1-8)", 1, 8, 5)

# final prediction logic
if st.button("Calculate Risk Score", type="primary"):
    
    column_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]

    features = [
        high_bp_num, high_chol_num, int(chol_check), bmi, int(smoker), int(stroke), 
        int(heart_dis), int(phys_act), int(fruits), int(veggies), int(hvy_alc), 
        int(healthcare), int(doc_cost), gen_hlth, ment_hlth, phys_hlth, 
        int(diff_walk), (1 if sex == "Male" else 0), age_cat, edu, income
    ]

    # Create the DataFrame 
    features_df = pd.DataFrame([features], columns=column_names)

    # Scale and Predict 
    features_scaled = scaler.transform(features_df)
    prob = model.predict_proba(features_scaled)[0][1]
    
    st.divider()
    if prob > 0.5:
        st.error(f"### High Risk Identified: {prob:.1%}")
        st.write("Your health profile shows a strong correlation with diabetic indicators. Please consult a medical professional for a formal screening.")
    else:
        st.success(f"### Low Risk Identified: {1-prob:.1%} Confidence")
        st.write("Your indicators suggest you are currently in a lower risk category for diabetes.")

st.caption("Disclaimer: This is a statistical model for educational use and not a medical diagnosis.")