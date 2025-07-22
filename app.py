# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

# --- Page config ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔍", layout="wide")

# --- Logo Section ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    logo = Image.open("logo.png")
    resized_logo = logo.resize((600, 150))  # Adjust logo size for better fit
    st.image(resized_logo)

# --- App Title ---
st.markdown("<h1 style='text-align: center;font-size: 50px; color: #FFFFFF;'> Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Model and Preprocessing Objects ---
with open("churn_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # This must be a StandardScaler object, not a NumPy array

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# --- Input Form ---
st.markdown("### 📋 Enter Customer Details")

with st.form("churn_form"):
    col1, col2 = st.columns([1, 1])

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
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)

    submitted = st.form_submit_button(" Predict Churn")

# --- Prediction ---
if submitted:
    st.markdown("---")

    # Prepare input dictionary for prediction
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
    }

    # Add contract one-hot encoding
    input_dict[f'Contract_{Contract}'] = 1

    # Convert to DataFrame and align with model features
    input_df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Scale input and make prediction
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Display result
    st.markdown("###  Prediction Result:")
    if prediction == 1:
        st.error("❌ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
