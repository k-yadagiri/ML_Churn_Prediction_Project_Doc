import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(page_title="Customer Churn Prediction - Yadagiri", page_icon="🔍", layout="wide")

# --- Load Model & Scaler ---
MODEL_PATH = "advanced_churn_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["features"]
except FileNotFoundError:
    st.error(f"❌ File '{MODEL_PATH}' not found. Please ensure it's in the same directory as 'app.py'.")
    st.stop()
except KeyError as e:
    st.error(f"❌ Missing key in pickle file: {e}. Expected keys: 'model', 'scaler', 'features'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load the model file: {e}")
    st.stop()

# --- Logo Section ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        logo = Image.open("logo.png")
        resized_logo = logo.resize((600, 150))
        st.image(resized_logo)
    except:
        st.markdown("<h3 style='text-align:center;'>📊 Customer Churn Prediction App - Yadagiri</h3>", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;font-size: 50px; color: #FFFFFF;'>Customer Churn Prediction by Yadagiri</h1>", unsafe_allow_html=True)
st.markdown("---")

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

    # Create input dict
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

    # Ensure all feature columns exist
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.markdown("### 🎯 Prediction Result:")
        if prediction == 1:
            st.error("❌ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤ by Yadagiri</p>", unsafe_allow_html=True)
