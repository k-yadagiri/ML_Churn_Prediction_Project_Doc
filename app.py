import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# --- Page Config ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- Updated CSS for Light Theme with Dark Text ---
st.markdown("""
<style>
body, .stApp {
    background-color: #f2f5f9;
    font-family: 'Segoe UI', sans-serif;
    color: #1e1e1e;
}
h1, h2, h3, h4, h5, h6, label, .css-1cpxqw2 {
    color: #1e1e1e !important;
}
input, select, textarea {
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem !important;
    background: #e0e5ec !important;
    color: #1e1e1e !important;
    box-shadow: inset 5px 5px 10px #c2c9d6, inset -5px -5px 10px #ffffff;
}
.stButton>button {
    background-color: #dee4f1;
    color: #1e1e1e !important;
    padding: 0.6em 1em;
    border-radius: 12px;
    border: none;
    box-shadow: 5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #cfd7e5;
}
.gauge-card {
    background-color: #e0e5ec;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff;
    margin-top: 20px;
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
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

# Check model exists
if not os.path.exists("advanced_churn_model.pkl"):
    st.error("❌ Model file not found.")
    st.stop()

# Load
data = load_data()
model, scaler, model_columns = load_model()

# --- Header ---
st.title("📱 Telecom Churn Predictor")
st.write("Enter customer details to predict churn risk:")

st.divider()

# --- Form Input Section ---
with st.form("prediction_form"):
    st.subheader("🎛️ Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("📅 Tenure (Months)", 0, 100, 12)
        monthly = st.number_input("💰 Monthly Charges ($)", 0.0, 200.0, 70.0)
        total = st.number_input("💵 Total Charges ($)", 0.0, 10000.0, 2500.0)
    with col2:
        contract = st.selectbox("📄 Contract Type", ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox("💳 Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        internet = st.selectbox("🌐 Internet Service", ['DSL', 'Fiber optic', 'No'])

    submit = st.form_submit_button("🚀 Predict Churn")

# --- Prediction ---
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

    prediction = model.predict(scaler.transform(input_df))[0]
    prob = model.predict_proba(scaler.transform(input_df))[0][1] * 100

    # Gauge Chart
    st.markdown("<h4>📊 Churn Risk Gauge</h4>", unsafe_allow_html=True)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#5b8def"},
            'steps': [
                {'range': [0, 40], 'color': "#c8e6c9"},
                {'range': [40, 70], 'color': "#fff59d"},
                {'range': [70, 100], 'color': "#ef9a9a"}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # Risk Interpretation
    with st.container():
        st.markdown("<div class='gauge-card'>", unsafe_allow_html=True)
        if prob > 70:
            st.markdown(f"❌ **High Risk of Churn** - {prob:.1f}%")
            st.write("🔁 Consider proactive retention strategies.")
        elif prob > 40:
            st.markdown(f"⚠️ **Moderate Risk** - {prob:.1f}%")
            st.write("📞 Might require engagement.")
        else:
            st.markdown(f"✅ **Low Risk** - {prob:.1f}%")
            st.write("🎉 Customer likely to stay.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<br><hr style='border-color:#ccc'/>
<div style='text-align:center; color:gray'>
Made with 💙 by <a href='https://github.com/k-yadagiri' target='_blank'>Yadagiri</a>
</div>
""", unsafe_allow_html=True)
