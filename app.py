import streamlit as st
import pandas as pd
import joblib

page_style = """
<style>
/* Page background */
[data-testid="stAppViewContainer"] { background-color: #dbe9f4; }

/* Title and subtitle */
.title { 
    font-size: 38px; 
    font-weight: 800; 
    color: #1f3c88; 
    text-align: center; 
    margin-bottom: 5px; 
}

.subtitle { 
    font-size: 17px; 
    color: #444; 
    text-align: center; 
    margin-bottom: 25px; 
}

/* Input fields */
div[data-baseweb="input"] input {
    background-color: #e0f0ff !important;
    border: 2px solid #9fc9ff !important;
    border-radius: 6px !important;
    padding: 6px !important;
    font-weight: 600 !important;
    color: #1f3c88 !important;
}

/* Dropdowns */
div[data-baseweb="select"] div[role="combobox"] {
    background-color: #e0f0ff !important;
    border: 2px solid #9fc9ff !important;
    border-radius: 6px !important;
    padding: 6px 30px 6px 6px !important; /* leave space for arrow */
    font-weight: 600 !important;
    color: #1f3c88 !important;
    position: relative;
}

/* Native arrow for dropdowns */
div[data-baseweb="select"] div[role="combobox"]::after {
    content: "▼";        /* standard dropdown arrow */
    color: #1f3c88;     
    font-size: 14px;    
    position: absolute;  
    right: 10px;        
    top: 50%;           
    transform: translateY(-50%);
    pointer-events: none;
}

/* Prediction box */
.pred-box { 
    padding: 18px; 
    border-radius: 12px; 
    text-align: center; 
    font-size: 22px; 
    font-weight: 700; 
    margin-top: 20px; 
}

.good { 
    background-color: #b6e2ff; 
    color: #045275; 
    border: 2px solid #045275; 
}

.bad { 
    background-color: #ffd6d6; 
    color: #8b1a1a; 
    border: 2px solid #8b1a1a; 
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

st.markdown(page_style, unsafe_allow_html=True)

#Load Model & Encoders
model = joblib.load("best_credit_model.pkl")
encoder_files = {
    "Sex": "Sex_encoder.pkl",
    "Housing": "Housing_encoder.pkl",
    "Saving accounts": "Saving accounts_encoder.pkl",
    "Checking account": "Checking account_encoder.pkl"
}
encoders = {col: joblib.load(file) for col, file in encoder_files.items()}
target_encoder = joblib.load("target_encoder.pkl")

#Header
st.markdown('<div class="title">Credit Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fill in the details below to predict credit risk</div>', unsafe_allow_html=True)

#Form for Inputs
with st.form("credit_form"):

    age = st.number_input("Age", min_value=18, max_value=75, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.number_input("Duration (months)", min_value=1, value=12)

    submitted = st.form_submit_button("Predict Risk")

#Prediction
if submitted:
    try:
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [job],
            "Housing": [encoders["Housing"].transform([housing])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })

        pred = model.predict(input_df)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]

        if pred_label.lower() == "good":
            st.markdown('<div class="pred-box good">Prediction: GOOD Credit Risk</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pred-box bad">Prediction: BAD Credit Risk</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
