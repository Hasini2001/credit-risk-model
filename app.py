import streamlit as st
import pandas as pd
import joblib

#Page Style
page_style = """
<style>
[data-testid="stAppViewContainer"] { background-color: #dbe9f4; }

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

div[data-baseweb="input"] input {
    background-color: #e0f0ff !important;
    border: 2px solid #9fc9ff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    color: #1f3c88 !important;
}

div[data-baseweb="select"] div[role="combobox"] {
    background-color: #e0f0ff !important;
    border: 2px solid #9fc9ff !important;
    border-radius: 6px !important;
    padding-right: 30px !important;
    font-weight: 600 !important;
    color: #1f3c88 !important;
    position: relative;
}

div[data-baseweb="select"] div[role="combobox"]::after {
    content: "▼";
    color: #1f3c88;
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
}

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

#Load Model & Encoders
model = joblib.load("best_credit_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")  # saved from training

#Header
st.markdown('<div class="title">Credit Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fill in the details below to predict credit risk</div>',
    unsafe_allow_html=True
)

#Input Form
with st.form("credit_form"):

    age = st.number_input("Age", min_value=18, max_value=75, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving_accounts = st.selectbox(
        "Saving Accounts", ["little", "moderate", "quite rich", "rich"]
    )
    checking_account = st.selectbox(
        "Checking Account", ["little", "moderate", "rich"]
    )
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.number_input("Duration (months)", min_value=1, value=12)

    submitted = st.form_submit_button("Predict Risk")

#Predictions
if submitted:
    try:
        # Raw input dataframe (same as training BEFORE encoding)
        input_raw = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "Job": [job],
            "Housing": [housing],
            "Saving accounts": [saving_accounts],
            "Checking account": [checking_account],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })

        # One-hot encode (same method used in training)
        input_encoded = pd.get_dummies(input_raw, drop_first=True)

        # Align columns with training data
        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Prediction
        pred = model.predict(input_encoded)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]

        # Output
        if pred_label.lower() == "good":
            st.markdown(
                '<div class="pred-box good">Prediction: GOOD Credit Risk</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="pred-box bad">Prediction: BAD Credit Risk</div>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
