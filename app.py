import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(page_title="Customer Churn Prediction - Yadagiri", page_icon="🔍", layout="wide")

# --- Theme Colors ---
primary_bg = "#0d1117"
card_bg = "#161b22"
text_color = "#f0f6fc"
accent = "#00e1ff"
shadow = "0px 0px 10px rgba(0,225,255,0.5)"

# --- Custom CSS ---
st.markdown(f"""
<style>
body, .stApp {{
    background-color: {primary_bg};
    color: {text_color};
}}
.big-title {{
    font-size:32px !important;
    font-weight:bold;
    color: {accent};
}}
.subtitle {{
    font-size:18px;
    color: #9ca3af;
    margin-bottom: 20px;
}}
.metric-card {{
    background-color: {card_bg};
    padding:20px;
    border-radius:12px;
    box-shadow: 0 0 10px rgba(0,225,255,0.2);
    text-align:center;
}}
.pred-card {{
    background-color: {card_bg};
    padding:15px 20px;
    border-radius:12px;
    margin-bottom:10px;
    border: 1px solid {accent};
    box-shadow: {shadow};
}}
.result-card {{
    background-color: {card_bg};
    padding:15px 20px;
    border-radius:12px;
    margin-top:20px;
    border: 1px solid {accent};
    box-shadow: {shadow};
}}
.footer {{
    color: gray;
    text-align: center;
    font-size: 13px;
    margin-top: 40px;
}}
a.footer-link {{
    color: #9ca3af;
    text-decoration: none;
}}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Branding ---
with st.sidebar:
    st.markdown("## Yadagiri")
    st.markdown("Customer Churn Prediction App")
    st.markdown("[🔗 View on GitHub](https://github.com/k-yadagiri/churn_prediction)")

# --- Load Data & Model ---
@st.cache_data
def load_data():
    return pd.read_csv("churn_dataset.csv")

@st.cache_resource
def load_model():
    with open("advanced_churn_model.pkl", "rb") as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

# Check if model file exists
if not os.path.exists("advanced_churn_model.pkl"):
    st.error("❌ Model file not found. Please ensure 'advanced_churn_model.pkl' is in the app directory.")
    st.stop()

# Load everything
data = load_data()
model, scaler, model_columns = load_model()

# --- Header ---
st.markdown("<div class='big-title'>📊 Telecom Customer Churn Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Understand why customers churn & predict risk instantly.</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Metric Cards ---
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>📉 Churn Rate</h4><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>👥 Total Customers</h4><h2>{len(data):,}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>💲 Avg Monthly</h4><h2>${data['MonthlyCharges'].mean():.2f}</h2></div>", unsafe_allow_html=True)

# --- Tabs for Navigation ---
tab1, tab2 = st.tabs(["🏠 Prediction", "📊 Insights"])

# --- Tab 1: Prediction Form ---
with tab1:
    st.markdown("<h2 style='color:#00e1ff;'>🔮 Predict Customer Churn</h2>", unsafe_allow_html=True)
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider('📅 Tenure (months)', 0, 100, 12)
            monthly = st.number_input('💰 Monthly Charges ($)', 0.0, 200.0, 70.0)
            total = st.number_input('💵 Total Charges ($)', 0.0, 10000.0, 2500.0)
        with c2:
            contract = st.selectbox('📄 Contract Type', ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox('💳 Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            internet = st.selectbox('🌐 Internet Service', ['DSL', 'Fiber optic', 'No'])

        predict_btn = st.form_submit_button("🚀 Predict Churn")

    if predict_btn:
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
        pred = model.predict(scaler.transform(input_df))[0]
        prob = model.predict_proba(scaler.transform(input_df))[0][1]*100

        st.markdown("<div class='result-card'><h4>📊 Prediction Result</h4>", unsafe_allow_html=True)
        if prob > 70:
            st.markdown(f"<div class='result-card'>❌ <strong>High churn risk!</strong> Estimated risk: <b>{prob:.1f}%</b>.<br>"
                        f"👉 Suggest loyalty discount or proactive contact.</div>", unsafe_allow_html=True)
        elif prob > 40:
            st.markdown(f"<div class='result-card'>⚠ <strong>Medium churn risk:</strong> <b>{prob:.1f}%</b>.<br>"
                        f"👉 Consider engagement strategies.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card'>✅ <strong>Low churn risk:</strong> <b>{prob:.1f}%</b>.<br>"
                        f"Customer likely to stay.</div>", unsafe_allow_html=True)

# --- Tab 2: Insights ---
with tab2:
    st.markdown("<h2 style='color:#00e1ff;'>📊 Data Insights & Exploratory Analysis</h2>", unsafe_allow_html=True)

    st.subheader("✅ Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn', color_discrete_sequence=['#FF6B6B','#4ECDC4'])
    fig1.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💳 Churn by Payment Method")
    churn_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes'] * 100
    fig2 = px.bar(churn_payment.sort_values(), orientation='h', color=churn_payment, color_continuous_scale='blues')
    fig2.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📑 Churn by Contract Type")
    churn_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes'] * 100
    fig3 = px.bar(x=churn_contract.index, y=churn_contract.values, color=churn_contract.values, color_continuous_scale='teal')
    fig3.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig3, use_container_width=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    <hr style='border-color:#2f3b48;'/>
    <p>💡 Developed with ❤️ by <strong>Yadagiri</strong> | 
    <a class='footer-link' href='https://github.com/k-yadagiri' target='_blank'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
