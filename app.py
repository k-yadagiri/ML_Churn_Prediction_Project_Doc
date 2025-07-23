import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# --- Page Config ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- Light Theme & Neumorphic CSS ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
body, .stApp {
    background-color: #f2f5f9;
    font-family: 'Poppins', sans-serif;
    color: #1b1b1b;
}

h1, h2, h3, h4, h5, h6, label, .stTextInput label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #111 !important;
    font-weight: 600 !important;
}

input, select, textarea {
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem !important;
    background: #e0e5ec !important;
    box-shadow: inset 5px 5px 10px #c2c9d6, inset -5px -5px 10px #ffffff;
    color: #1a1a1a !important;
    font-weight: 500 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #e0e5ec !important;
    color: #1a1a1a !important;
}

.stNumberInput input {
    background-color: #e0e5ec !important;
    color: #1a1a1a !important;
    font-weight: 500 !important;
}

.stSlider > div {
    color: #1a1a1a !important;
    font-weight: 500;
}

.stButton > button {
    background-color: #dee4f1 !important;
    color: #1e1e1e !important;
    padding: 0.6em 1.2em !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #cfd7e5 !important;
    color: #000 !important;
}

.metric-box {
    background-color: #e0e5ec;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff;
    text-align: center;
    color: #212121;
    font-weight: 600;
}

.gauge-card {
    background-color: #e0e5ec;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff;
    margin-top: 20px;
    color: #1f1f1f;
    font-weight: 500;
}

/* Fix dropdown and select visibility */
[data-baseweb="select"] {
    background-color: #e0e5ec !important;
    color: #1a1a1a !important;
    font-weight: 500 !important;
}

.stMarkdown p {
    color: #1b1b1b !important;
}

hr {
    border-color: #ccc;
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

# --- Metric Cards ---
churn_rate = (data['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'><h5>Churn Rate</h5><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><h5>Total Customers</h5><h2>{len(data):,}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><h5>Avg Monthly Charges</h5><h2>${data['MonthlyCharges'].mean():.2f}</h2></div>", unsafe_allow_html=True)

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

    # Gauge Chart (simulate neumorphic circular meter)
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
