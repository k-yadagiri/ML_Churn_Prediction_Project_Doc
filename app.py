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
    try:
        logo = Image.open("logo.png")
        resized_logo = logo.resize((600, 150))
        st.image(resized_logo)
    except:
        st.write("")

# --- App Title ---
st.markdown("<h1 style='text-align: center;font-size: 50px; color: #FFFFFF;'> Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load model ---
try:
    with open("/mnt/data/advanced_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load the model file. Please ensure the .pkl model is correctly uploaded.")
    st.stop()

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

    submitted = st.form_submit_button("🔍 Predict Churn")

# --- Prediction ---
if submitted:
    st.markdown("---")

    # Prepare input dictionary
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

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Align with model's expected features
    try:
        model_features = model.feature_names_in_
    except AttributeError:
        model_features = input_df.columns  # fallback if model doesn't provide feature names

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    # Predict
    prediction = model.predict(input_df)[0]

    # Display result
    st.markdown("### 🎯 Prediction Result:")
    if prediction == 1:
        st.error("❌ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
