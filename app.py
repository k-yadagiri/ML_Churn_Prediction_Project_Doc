import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="", layout="wide")

# Dark theme colors
primary_bg = "#0d1117"
card_bg = "#161b22"
text_color = "#f0f6fc"
accent = "#00e1ff"
shadow = "0px 0px 10px rgba(0,225,255,0.5)"

# Custom CSS
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
.big-btn > button {{
    background-color: {accent};
    color: black;
    width: 100%;
    padding: 0.75em;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
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

# Sidebar branding
with st.sidebar:
    st.markdown("Yadagiri")
    st.markdown("Customer Churn Prediction App")
    st.markdown("[View on GitHub](https://github.com/k-yadagiri/churn_prediction)")

# Load data & model
@st.cache_data
def load_data():
    return pd.read_csv('churn_dataset.csv')

@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

data = load_data()
model, scaler, model_columns = load_model()

# Sidebar navigation (move AFTER load to avoid NameError)
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Insights"])

# Header
st.markdown(f"<div class='big-title'>Telecom Customer Churn Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Understand why customers churn & predict risk instantly.</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Metrics
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>📉 Churn Rate</h4><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>👥 Total Customers</h4><h2>{len(data):,}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>💲 Avg Monthly</h4><h2>${data['MonthlyCharges'].mean():.2f}</h2></div>", unsafe_allow_html=True)

# Pages
if page == "🏠 Home":
    st.subheader("🔮 Predict if customer will churn")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='pred-card'>📅 Tenure (months)</div>", unsafe_allow_html=True)
            tenure = st.slider('Tenure (months)', 0, 100, 12)

            st.markdown("<div class='pred-card'>💰 Monthly Charges</div>", unsafe_allow_html=True)
            monthly = st.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)

            st.markdown("<div class='pred-card'>💵 Total Charges</div>", unsafe_allow_html=True)
            total = st.number_input('Total Charges ($)', 0.0, 10000.0, 2500.0)

        with c2:
            st.markdown("<div class='pred-card'>📄 Contract Type</div>", unsafe_allow_html=True)
            contract = st.selectbox('Select Contract Type', ['Month-to-month', 'One year', 'Two year'])

            st.markdown("<div class='pred-card'>💳 Payment Method</div>", unsafe_allow_html=True)
            payment = st.selectbox('Select Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

            st.markdown("<div class='pred-card'>🌐 Internet Service</div>", unsafe_allow_html=True)
            internet = st.selectbox('Select Internet Service', ['DSL', 'Fiber optic', 'No'])

        predict_btn = st.form_submit_button("🚀 Predict Customer Churn Risk")

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
            st.markdown(f"<div class='result-card'>⚠ High churn risk! Estimated risk: {prob:.1f}.<br>"
                        f"👉 Customer likely to churn. Consider loyalty discount or proactive contact.</div>", unsafe_allow_html=True)
        elif prob > 40:
            st.markdown(f"<div class='result-card'>⚠ Medium churn risk: {prob:.1f}.<br>"
                        f"👉 Consider engagement strategies.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card'>✅ Low churn risk: {prob:.1f}.<br>"
                        f"Customer likely to stay. Continue current retention approach.</div>", unsafe_allow_html=True)

elif page == "📊 Insights":
    st.subheader("📊 Data Insights & EDA")
    st.subheader("✅ Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn', color_discrete_sequence=['#FF6B6B','#4ECDC4'])
    fig1.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💳 Churn by Payment Method")
    churn_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig2 = px.bar(churn_payment.sort_values(), orientation='h', color=churn_payment, color_continuous_scale='blues')
    fig2.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📑 Churn by Contract Type")
    churn_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig3 = px.bar(x=churn_contract.index, y=churn_contract.values, color=churn_contract.values, color_continuous_scale='teal')
    fig3.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("<div class='footer'>Developed by Yadagiri | "
            "<a class='footer-link' href='https://github.com/k-yadagiri'>GitHub</a></div>", 
            unsafe_allow_html=True)
