# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os # Import os module for path manipulation

# --- Page config ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔍", layout="wide")

# --- Logo Section ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Define the path to your logo. Assumes 'logo.png' is in the same directory as this script.
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        resized_logo = logo.resize((600, 150))  # Adjust logo size for better fit
        st.image(resized_logo)
    else:
        st.warning(f"Logo file not found at {logo_path}. Please ensure 'logo.png' is in the correct directory.")

# --- App Title ---
st.markdown("<h1 style='text-align: center;font-size: 50px; color: #FFFFFF;'> Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Model ---
# Define path to your model file.
model_file = "advanced_churn_model.pkl"

try:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    st.error(f"Error loading the model file: '{model_file}'. "
             "Please ensure the model file is in the correct directory.")
    st.stop() # Stop the app execution if the crucial model file is not found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}. "
             "The model file might be corrupted or in an incompatible format.")
    st.stop() # Stop the app execution if model loading fails

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
    # NOTE: The order of these features MUST exactly match the order
    # the model was trained on, and all expected features must be present.
    # Without feature_names.pkl and the alignment loop, this is critical.
    input_dict = {
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        # One-hot encoded categorical features
        # Ensure these are exactly how they appeared during training
        # For example: 'gender_Male' or 'gender_Female'
        'gender_Female': 1 if gender == 'Female' else 0,
        'gender_Male': 1 if gender == 'Male' else 0,
        'Partner_No': 1 if Partner == 'No' else 0,
        'Partner_Yes': 1 if Partner == 'Yes' else 0,
        'Dependents_No': 1 if Dependents == 'No' else 0,
        'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
        'PhoneService_No': 1 if PhoneService == 'No' else 0,
        'PhoneService_Yes': 1 if PhoneService == 'Yes' else 0,
        'MultipleLines_No': 1 if MultipleLines == 'No' else 0,
        'MultipleLines_No phone service': 1 if MultipleLines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,
        'InternetService_DSL': 1 if InternetService == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'OnlineSecurity_No': 1 if OnlineSecurity == 'No' else 0,
        'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
        'OnlineBackup_No': 1 if OnlineBackup == 'No' else 0,
        'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0,
        'DeviceProtection_No': 1 if DeviceProtection == 'No' else 0,
        'DeviceProtection_Yes': 1 if DeviceProtection == 'Yes' else 0,
        'TechSupport_No': 1 if TechSupport == 'No' else 0,
        'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
        'StreamingTV_No': 1 if StreamingTV == 'No' else 0,
        'StreamingTV_Yes': 1 if StreamingTV == 'Yes' else 0,
        'Contract_Month-to-month': 1 if Contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
    }

    # Convert to DataFrame
    # Note: Without 'feature_names', the column order here MUST exactly match
    # the order of features used during model training.
    # It's generally safer to ensure consistent column order if not using a feature_names list.
    # For now, we'll convert the dictionary values to a list in a pre-defined order.
    # This requires you to know the exact feature order from your training!
    # A safer approach for exact feature ordering is complex without feature_names.pkl
    # For simplicity, if your model was trained on a specific DataFrame column order,
    # you'd need to manually create that exact column order here for input_df.

    # Instead of dynamically creating columns based on a dict,
    # let's create a DataFrame with all possible one-hot encoded columns
    # initialized to 0, then set the selected ones to 1.
    # This is a manual re-creation of the feature_names logic.

    # THIS LIST OF 'all_possible_features' MUST EXACTLY MATCH THE ORDER AND NAMES
    # OF THE FEATURES YOUR MODEL WAS TRAINED ON. IF IT DOESN'T, YOUR PREDICTIONS
    # WILL BE INCORRECT OR THE APP WILL CRASH.
    all_possible_features = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges',
        'gender_Female', 'gender_Male',
        'Partner_No', 'Partner_Yes',
        'Dependents_No', 'Dependents_Yes',
        'PhoneService_No', 'PhoneService_Yes',
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year'
    ]

    # Create an empty DataFrame with all possible columns
    input_df = pd.DataFrame(0, index=[0], columns=all_possible_features)

    # Populate the DataFrame with user input
    input_df['SeniorCitizen'] = SeniorCitizen
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = MonthlyCharges

    # Set selected one-hot encoded values
    input_df[f'gender_{gender}'] = 1
    input_df[f'Partner_{Partner}'] = 1
    input_df[f'Dependents_{Dependents}'] = 1
    input_df[f'PhoneService_{PhoneService}'] = 1
    input_df[f'MultipleLines_{MultipleLines}'] = 1
    input_df[f'InternetService_{InternetService}'] = 1
    input_df[f'OnlineSecurity_{OnlineSecurity}'] = 1
    input_df[f'OnlineBackup_{OnlineBackup}'] = 1
    input_df[f'DeviceProtection_{DeviceProtection}'] = 1
    input_df[f'TechSupport_{TechSupport}'] = 1
    input_df[f'StreamingTV_{StreamingTV}'] = 1
    input_df[f'Contract_{Contract}'] = 1


    # Make prediction - NO SCALING APPLIED HERE
    prediction = model.predict(input_df)[0]

    # Display result
    st.markdown("###  Prediction Result:")
    if prediction == 1:
        st.error("❌ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
