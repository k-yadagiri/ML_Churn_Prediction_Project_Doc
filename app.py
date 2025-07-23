import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os

# --- Page Config ---
st.set_page_config(page_title="Churn Prediction & EDA", layout="wide")

# --- Custom Dark Theme CSS ---
st.markdown("""
<style>
body, .stApp {
    background-color: #0e1117;
    color: #ffffff;
}
h1, h2, h3, h4, h5, h6, label, .css-1cpxqw2 {
    color: #ffffff !important;
}
input, select, textarea {
    border-radius: 8px !important;
    background-color: #1c1f26 !important;
    color: #ffffff !important;
    border: none !important;
}
.stButton>button {
    background-color: #1f6feb;
    color: white;
    font-weight: bold;
    padding: 0.5em 1em;
    border-radius: 8px;
    border: none;
}
.stButton>button:hover {
    background-color: #388bfd;
}
.gauge-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    return pd.read_csv("churn_dataset.csv")

@st.cache_resource
def load_model():
    with open("advanced_churn_model.pkl", "rb") as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

if not os.path.exists("advanced_churn_model.pkl"):
    st.error("❌ Model file not found.")
    st.stop()

df = load_data()
model, scaler, model_columns = load_model()

# --- Navigation ---
st.sidebar.title("📊 Menu")
option = st.sidebar.radio("Go to", ["EDA Dashboard", "Churn Prediction"])

# === EDA DASHBOARD ===
if option == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧑 Gender Distribution")
        gender_fig = px.histogram(df, x="gender", color="gender", color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(gender_fig, use_container_width=True)

    with col2:
        st.subheader("📄 Contract Types")
        contract_fig = px.pie(df, names="Contract", color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(contract_fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("💰 Monthly Charges Distribution")
        monthly_fig = px.histogram(df, x="MonthlyCharges", nbins=30, color_discrete_sequence=["#636efa"])
        st.plotly_chart(monthly_fig, use_container_width=True)

    with col4:
        st.subheader("⚠️ Churn Rate")
        churn_fig = px.pie(df, names="Churn", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(churn_fig, use_container_width=True)

# === PREDICTION PAGE ===
if option == "Churn Prediction":
    st.title("📱 Predict Customer Churn")

    with st.form("form"):
        st.subheader("🎛️ Input Customer Info")

        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("📅 Tenure (Months)", 0, 100, 12)
            monthly = st.number_input("💰 Monthly Charges", 0.0, 200.0, 70.0)
            total = st.number_input("💵 Total Charges", 0.0, 10000.0, 2500.0)
        with col2:
            contract = st.selectbox("📄 Contract", ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox("💳 Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            internet = st.selectbox("🌐 Internet Service", ['DSL', 'Fiber optic', 'No'])

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_df = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            f'Contract_{contract}': [1],
            f'PaymentMethod_{payment}': [1],
            f'InternetService_{internet}': [1]
        })

        for col in model_columns:
            if col not in input_df:
                input_df[col] = 0
        input_df = input_df[model_columns]

        prob = model.predict_proba(scaler.transform(input_df))[0][1] * 100

        st.subheader("📊 Prediction Result")
        st.markdown("<div class='gauge-card'>", unsafe_allow_html=True)

        if prob > 70:
            st.error(f"❌ High Churn Risk - {prob:.2f}%")
        elif prob > 40:
            st.warning(f"⚠️ Medium Churn Risk - {prob:.2f}%")
        else:
            st.success(f"✅ Low Churn Risk - {prob:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<hr style='border-color:#333'/>
<div style='text-align:center; color:#888'>
Made with ❤️ by <a href='https://github.com/k-yadagiri' style='color:#ccc' target='_blank'>Yadagiri</a>
</div>
""", unsafe_allow_html=True)
