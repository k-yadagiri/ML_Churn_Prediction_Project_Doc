import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
h1, h2, h3, h4, h5, h6, label {
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
</style>
""", unsafe_allow_html=True)

# --- Load Data & Model ---
@st.cache_data
def load_data():
    return pd.read_csv("churn_dataset.csv")

@st.cache_resource
def load_model():
    with open("advanced_churn_model.pkl", "rb") as f:
        model, scaler, model_columns = pickle.load(f)
    return model, scaler, model_columns

if not os.path.exists("advanced_churn_model.pkl"):
    st.error("❌ Model file not found.")
    st.stop()

df = load_data()
model, scaler, model_columns = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("📊 Menu")
option = st.sidebar.radio("Go to", ["EDA Dashboard", "Churn Prediction"])

# === EDA DASHBOARD ===
if option == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧑 Gender Distribution")
        fig1 = px.histogram(df, x="gender", color="gender", color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📄 Contract Type")
        fig2 = px.pie(df, names="Contract", color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("💰 Monthly Charges")
        fig3 = px.histogram(df, x="MonthlyCharges", nbins=30, color_discrete_sequence=["#636efa"])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("⚠️ Churn Rate")
        fig4 = px.pie(df, names="Churn", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig4, use_container_width=True)

# === CHURN PREDICTION ===
elif option == "Churn Prediction":
    st.title("📱 Predict Customer Churn")

    with st.form("input_form"):
        st.subheader("📝 Input Customer Information")

        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (in months)", 0, 100, 12)
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)
        with col2:
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        input_data = {
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            f'Contract_{contract}': [1],
            f'PaymentMethod_{payment}': [1],
            f'InternetService_{internet}': [1]
        }

        for col in model_columns:
            if col not in input_data:
                input_data[col] = [0]
        input_df = pd.DataFrame(input_data)
        input_df = input_df[model_columns]

        prob = model.predict_proba(scaler.transform(input_df))[0][1] * 100

        st.subheader("🎯 Churn Result")
        if prob > 70:
            st.error(f"❌ High Churn Risk: {prob:.2f}%")
        elif prob > 40:
            st.warning(f"⚠️ Medium Churn Risk: {prob:.2f}%")
        else:
            st.success(f"✅ Low Churn Risk: {prob:.2f}%")

        st.subheader("📟 Churn Probability Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#1f6feb"},
                'steps': [
                    {'range': [0, 40], 'color': "#00cc96"},
                    {'range': [40, 70], 'color': "#ffa600"},
                    {'range': [70, 100], 'color': "#ef553b"},
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0e1117",
            font={'color': "white", 'family': "Arial"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Footer ---
st.markdown("""
<hr style='border-color:#333'/>
<div style='text-align:center; color:#888'>
Made with ❤️ by <a href='https://github.com/k-yadagiri' style='color:#ccc' target='_blank'>Yadagiri</a>
</div>
""", unsafe_allow_html=True)
