import streamlit as st 
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(page_title="Churn Predictor", layout="wide")

# --- Dark Mode Toggle & Theme Variables ---
components.html("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg-color: #f2f5f9;
    --text-color: #1b1b1b;
    --input-bg: #e0e5ec;
    --box-shadow: 5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff;
}
body {
    font-family: 'Poppins', sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
}
button#toggle-theme {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 9999;
    padding: 6px 16px;
    background: #5b8def;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    cursor: pointer;
}
</style>
<button id="toggle-theme">🌓 Toggle Theme</button>
<script>
const root = document.documentElement;
const toggleBtn = document.getElementById("toggle-theme");

toggleBtn.onclick = () => {
    const currentBg = getComputedStyle(root).getPropertyValue('--bg-color').trim();
    if (currentBg === '#f2f5f9') {
        root.style.setProperty('--bg-color', '#1c1c1c');
        root.style.setProperty('--text-color', '#f5f5f5');
        root.style.setProperty('--input-bg', '#2a2a2a');
        root.style.setProperty('--box-shadow', '5px 5px 15px #111, -5px -5px 15px #2e2e2e');
        document.body.style.backgroundColor = '#1c1c1c';
        document.body.style.color = '#f5f5f5';
    } else {
        root.style.setProperty('--bg-color', '#f2f5f9');
        root.style.setProperty('--text-color', '#1b1b1b');
        root.style.setProperty('--input-bg', '#e0e5ec');
        root.style.setProperty('--box-shadow', '5px 5px 15px #c2c9d6, -5px -5px 15px #ffffff');
        document.body.style.backgroundColor = '#f2f5f9';
        document.body.style.color = '#1b1b1b';
    }
};
</script>
""", height=0)

# --- Theme Styling ---
st.markdown("""
<style>
body, .stApp {
    background-color: var(--bg-color) !important;
    font-family: 'Poppins', sans-serif;
    color: var(--text-color) !important;
}
h1, h2, h3, h4, h5, h6, label, .stTextInput label, .stSelectbox label, .stSlider label {
    color: var(--text-color) !important;
    font-weight: 600 !important;
}
input, select, textarea {
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem !important;
    background: var(--input-bg) !important;
    box-shadow: inset 5px 5px 10px rgba(0,0,0,0.1), inset -5px -5px 10px rgba(255,255,255,0.7);
    color: var(--text-color) !important;
}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
}
.stButton > button {
    background-color: #dee4f1;
    color: var(--text-color);
    box-shadow: var(--box-shadow);
    font-weight: bold;
}
.metric-box, .gauge-card {
    background-color: var(--input-bg);
    box-shadow: var(--box-shadow);
    color: var(--text-color);
    border-radius: 12px;
    padding: 15px;
    font-weight: 600;
}
hr {
    border-color: #ccc;
}
@media screen and (max-width: 768px) {
    .metric-box, .gauge-card {
        font-size: 0.9rem;
        padding: 10px;
    }
    .stButton > button {
        width: 100%;
    }
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
