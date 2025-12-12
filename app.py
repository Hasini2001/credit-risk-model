import streamlit as st
import pandas as pd
import joblib

# ------------------ Custom Page Style ------------------
page_style = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #eef2f7;
}

/* Title */
.title {
    font-size: 38px;
    font-weight: 800;
    color: #1f3c88;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    font-size: 17px;
    color: #444;
    text-align: center;
    margin-bottom: 25px;
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
    background-color: #d6f8d6;
    color: #1b6d2b;
    border: 2px solid #1b6d2b;
}

.bad {
    background-color: #ffd6d6;
    color: #8b1a1a;
    border: 2px solid #8b1a1a;
}

/* ------------------ Uniform style for labels and inputs ------------------ */

/* Number inputs */
div[data-baseweb="input"] > label,
div[data-baseweb="input"] input {
    font-size: 16px !important;
    color: #1f3c88 !important;
    font-weight: 600 !important;
}

/* Selectboxes */
div[data-baseweb="select"] > label,
div[data-baseweb="select"] div[role="combobox"] {
    font-size: 16px !important;
    color: #1f3c88 !important;
    font-weight: 600 !important;
}

/* Input boxes background and padding */
div[data-baseweb="input"] input,
div[data-baseweb="select"] div[role="combobox"] {
    background-color: #f7f9fc !important;
    border-radius: 6px !important;
    padding: 6px !important;
}

/* Cursor pointer */
div[data-baseweb="select"],
div[data-baseweb="select"] * {
    cursor: pointer !important;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ------------------ Load Model & Encoders ------------------
model = joblib.load("xgb_credit_model.pkl")

columns = {
    "Sex": "Sex_encoder.pkl",
    "Housing": "Housing_encoder.pkl",
    "Saving accounts": "Saving accounts_encoder.pkl",
    "Checking accounts": "Checking account_encoder.pkl"
}
encode = {col: joblib.load(filename) for col, filename in columns.items()}

# ------------------ Header ------------------
st.markdown('<div class="title">Credit Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fill the details below</div>', unsafe_allow_html=True)

# ------------------ Inputs ------------------
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich"])
checking_accounts = st.selectbox("Checking Accounts", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, max_value=12, value=12)

# ------------------ Predict ------------------
if st.button("Predict Risk"):
    # Validate duration
    if duration < 1 or duration > 12:
        st.error("⚠ Duration must be between 1 and 12 months!")
    else:
        # Build input DataFrame
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encode["Sex"].transform([sex])[0]],
            "Job": [job],
            "Housing": [encode["Housing"].transform([housing])[0]],
            "Saving accounts": [encode["Saving accounts"].transform([saving_accounts])[0]],
            "Checking accounts": [encode["Checking accounts"].transform([checking_accounts])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })
        input_df = input_df.rename(columns={"Checking accounts": "Checking account"})

        # Make prediction
        pred = model.predict(input_df)[0]
        if pred == 1:
            st.markdown('<div class="pred-box good">Prediction: GOOD Credit Risk</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pred-box bad">Prediction: BAD Credit Risk</div>', unsafe_allow_html=True)
