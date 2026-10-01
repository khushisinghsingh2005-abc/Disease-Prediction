import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Disease Risk Assessment App",
    page_icon="🩺",
    layout="centered"
)

# Load Saved Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# UI Header
st.title("🩺 Health & Disease Risk Assessment")
st.markdown("Enter your health parameters below to check the risk prediction using our Machine Learning model.")

st.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.subheader("Patient Clinical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 90, 45)
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 24.5)
        bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
        
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", 50, 300, 100)
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 75)
        gender = st.selectbox("Gender", ["Female", "Male"])
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        exercise = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])

    submit_button = st.form_submit_button(label="Predict Disease Risk")

# Prediction Logic
if submit_button:
    # Preprocess inputs matching training encoding
    gender_val = 1 if gender == "Male" else 0
    smoking_val = 1 if smoking == "Yes" else 0
    exercise_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    exercise_val = exercise_mapping[exercise]
    
    # Create DataFrame for input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_val],
        'BMI': [bmi],
        'Blood_Pressure_mmHg': [bp],
        'Cholesterol_mg_dL': [cholesterol],
        'Glucose_mg_dL': [glucose],
        'Heart_Rate_bpm': [heart_rate],
        'Smoking': [smoking_val],
        'Exercise_Level': [exercise_val]
    })
    
    # Scale numerical columns
    num_cols = ['Age', 'BMI', 'Blood_Pressure_mmHg', 'Cholesterol_mg_dL', 'Glucose_mg_dL', 'Heart_Rate_bpm']
    input_data_scaled = input_data.copy()
    input_data_scaled[num_cols] = scaler.transform(input_data[num_cols])
    
    # Make Prediction
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1] * 100
    
    st.markdown("---")
    st.subheader("Assessment Results")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Disease Detected**")
        st.write(f"Model Confidence / Probability: **{probability:.2f}%**")
        st.warning("Recommendation: Please consult a medical professional for a detailed health checkup.")
    else:
        st.success(f"✅ **Low Risk / No Disease Predicted**")
        st.write(f"Disease Probability: **{probability:.2f}%**")
        st.info("Recommendation: Keep up the healthy lifestyle!")