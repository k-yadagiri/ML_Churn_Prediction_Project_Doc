import streamlit as st 
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os

# --- Page config ---
st.set_page_config(page_title="🔍 Churn Predictor", page_icon="🔍", layout="wide")

# --- Load Model ---
MODEL_PATH = "advanced_churn_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["features"]
except Exception as e:
    st.error("❌ Model file not found. Please ensure 'advanced_churn_model.pkl' is in the app directory.")
    st.stop()

# --- Header Banner ---
with st.container():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Predict if a customer will churn based on service usage</p>", unsafe_allow_html=True)
    st.markdown("---")

# --- Sidebar Info ---
with st.sidebar:
    st.header("📁 About")
    st.write("This app uses a machine learning model trained on telecom customer data to predict churn.")
    st.markdown("---")
    st.image("logo.png", use_column_width=True, caption="Built by Yadagiri")

# --- Input Section ---
st.subheader("📝 Enter Customer Details")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (in months)", 0, 72, 12)
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction ---
if submitted:
    input_dict = {
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        f'gender_{gender}': 1,
        f'Partner_{Partner}': 1,
        f'Dependents_{Dependents}': 1,
        f'PhoneService_{PhoneService}': 1,
        f'MultipleLines_{MultipleLines}': 1,
        f'InternetService_{InternetService}': 1,
        f'OnlineSecurity_{OnlineSecurity}': 1,
        f'OnlineBackup_{OnlineBackup}': 1,
        f'DeviceProtection_{DeviceProtection}': 1,
        f'TechSupport_{TechSupport}': 1,
        f'StreamingTV_{StreamingTV}': 1,
        f'Contract_{Contract}': 1
    }

    input_df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("## 🎯 Prediction Result:")
    if prediction == 1:
        st.error("❗ The customer is likely to **Churn**.")
    else:
        st.success("✅ The customer is likely to **Stay**.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Made with ❤️ by Yadagiri</p>", unsafe_allow_html=True)
