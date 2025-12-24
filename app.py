import streamlit as st
import pandas as pd
import joblib


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


model = joblib.load("best_credit_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")


st.markdown('<div class="title">Credit Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fill in the details below to predict credit risk</div>',
    unsafe_allow_html=True
)


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


if submitted:
    try:
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

        input_encoded = pd.get_dummies(input_raw, drop_first=True)
        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        pred = model.predict(input_encoded)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]

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
