import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Subscription Predictor", page_icon="📊", layout="centered")
st.title("Customer Subscription Prediction System")
st.write("Predict whether a customer is likely to subscribe using Logistic Regression (L2).")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "subscription_model.joblib")
scaler_path = os.path.join(BASE_DIR, "subscription_scaler.joblib")

@st.cache_resource
def load_artifacts():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.error("Model files are missing. Keep subscription_model.joblib and subscription_scaler.joblib in the same folder as app.py.")
    st.stop()

model, scaler = load_artifacts()

with st.form("customer_form"):
    st.subheader("Enter customer information")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=28)
    income = st.number_input("Monthly income (UGX)", min_value=0, value=1200000, step=50000)
    internet = st.number_input("Monthly internet usage (GB)", min_value=0.0, value=45.0, step=1.0)
    contacts = st.number_input("Previous marketing contacts", min_value=0, value=3, step=1)
    duration = st.number_input("Contact duration (minutes)", min_value=0.0, value=8.0, step=1.0)
    engagement = st.slider("Engagement score (1–10)", min_value=1, max_value=10, value=7)
    submitted = st.form_submit_button("Predict subscription")

if submitted:
    customer = pd.DataFrame([{
        "Age": age,
        "Monthly_Income": income,
        "Internet_Usage": internet,
        "Previous_Contacts": contacts,
        "Contact_Duration": duration,
        "Engagement_Score": engagement
    }])
    customer = customer[["Age", "Monthly_Income", "Internet_Usage", "Previous_Contacts", "Contact_Duration", "Engagement_Score"]]
    scaled = scaler.transform(customer)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0, 1])

    st.subheader("Prediction result")
    st.metric("Probability of subscription", f"{probability * 100:.1f}%")
    if prediction == 1:
        st.success("The model predicts that this customer is likely to subscribe.")
    else:
        st.warning("The model predicts that this customer is unlikely to subscribe.")
    st.caption("This is a model estimate, not a guarantee. Use predictions to support—not replace—human decisions.")
st.balloons()