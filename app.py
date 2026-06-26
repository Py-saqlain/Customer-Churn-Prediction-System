import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

st.set_page_config(page_title="Churn Prediction System", layout="centered")
st.title("Customer Churn Prediction System")
st.write("Input customer details below to predict the probability of churn.")

st.subheader("Demographics & Location")
age = st.slider("Age", 18, 70, 35)
gender = st.selectbox("Gender", ["Male", "Female"])
city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "Miami"])

st.subheader("Subscription & Usage")
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
tenure = st.slider("Tenure (Months Active)", 1, 60, 12)
monthly_spending = st.number_input("Monthly Spending ($)", 10.0, 100.0, 35.0)

st.subheader("Engagement Metrics")
num_purchases = st.slider("Number of Purchases", 0, 15, 3)
login_frequency = st.slider("Login Frequency (per month)", 1, 30, 10)
support_requests = st.slider("Customer Support Requests", 0, 10, 2)
satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 4)

if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Monthly_Spending": monthly_spending,
        "Tenure": tenure,
        "Number_of_Purchases": num_purchases,
        "Customer_Support_Requests": support_requests,
        "Login_Frequency": login_frequency,
        "Satisfaction_Score": satisfaction_score,
    }])
    
    for gen in ["Male"]:
        input_data[f"Gender_{gen}"] = 1 if gender == gen else 0
        
    for c in ["Houston", "Los Angeles", "Miami", "New York"]:
        input_data[f"City_{c}"] = 1 if city == c else 0
        
    for sub in ["Premium", "Standard"]:
        input_data[f"Subscription_Type_{sub}"] = 1 if subscription_type == sub else 0
        
    input_data = input_data.reindex(columns=model_features, fill_value=0)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f" **High Risk:** This customer is likely to churn! (Probability: {probability:.2%})")
    else:
        st.success(f" **Low Risk:** This customer is likely to stay. (Probability of leaving: {probability:.2%})")