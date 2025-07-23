import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# ---------- Page Setup ----------
st.set_page_config(page_title="Churn Prediction", layout="wide")

# ---------- Custom CSS for Light Neumorphic Theme ----------
st.markdown("""
    <style>
    body {
        background-color: #eaf1fb;
    }
    .main {
        background-color: #eaf1fb;
        color: #3d3d3d;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        background-color: #eaf1fb;
    }
    h1, h2, h3, h4, h5 {
        color: #3d3d3d;
    }
    label, .stRadio > label {
        color: #3d3d3d !important;
    }
    input, .stTextInput, .stSelectbox {
        background-color: #f4f7fd;
        border: 1px solid #cfd8e3;
        border-radius: 15px;
        color: #3d3d3d;
        padding: 8px 12px;
        box-shadow: 5px 5px 10px #d1d9e6, -5px -5px 10px #ffffff;
    }
    .stButton>button {
        background-color: #dfe9f3;
        color: #3d3d3d;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        box-shadow: 4px 4px 8px #c5ccd8, -4px -4px 8px #ffffff;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #d0def0;
        color: black;
        box-shadow: 2px 2px 6px #b0b8c0, -2px -2px 6px #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
model_path = "advanced_churn_model.pkl"
scaler_path = "scaler.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or Scaler file not found. Ensure they're in the app directory.")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("🔎 Churn Predictor")
st.sidebar.info("Fill in the user details to predict churn.")

# ---------- Main UI ----------
st.markdown("## 📋 Customer Information")
st.markdown("Please enter the following details:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, value=12)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# ---------- Prediction Button ----------
if st.button("🔍 Predict Churn"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "Contract": [contract],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "InternetService": [internet_service],
        "PaymentMethod": [payment_method]
    })

    # Encoding categorical data (Example only, match this with your preprocessing)
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer is likely to churn. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability: {1 - probability:.2%})")

# ---------- Footer ----------
st.markdown("---")
st.markdown("Made with ❤️ by **Yadagiri** | [GitHub](https://github.com/yadagiri)")
