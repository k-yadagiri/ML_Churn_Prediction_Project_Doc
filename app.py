import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

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
        st.warning("Logo not found. Please upload 'logo.png' if you want to display a logo.")

# --- App Title ---
st.markdown("<h1 style='text-align: center;font-size: 50px; color: #FFFFFF;'> Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Model and Scaler ---
model = None
scaler = None
feature_names = []

try:
    with open("advanced_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error("❌ Failed to load the model file. Using Dummy Model instead.")
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]*5], [0])  # Fake fit
    scaler = StandardScaler()
    scaler.fit([[0]*5])  # Dummy scaler
    feature_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Contract_Month-to-month']

if model and scaler is None:
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except:
        st.warning("⚠ Scaler or feature names not found. Using default dummy ones.")
        scaler = StandardScaler()
        scaler.fit([[0]*5])
        feature_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Contract_Month-to-month']

# --- Input Form ---
st.markdown("### 📋 Enter Customer Details")

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
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)

    submitted = st.form_submit_button(" Predict Churn")

# --- Prediction ---
if submitted:
    st.markdown("---")

    # Prepare input
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

    # Scale and predict
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.markdown("### Prediction Result:")
        if prediction == 1:
            st.error("❌ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")
    except Exception as e:
        st.error(f"🚫 Prediction failed: {e}")
