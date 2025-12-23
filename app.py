import streamlit as st
from backend import predict_credit_risk

#Page Style 
page_style = """
<style>
[data-testid="stAppViewContainer"] { background-color: #dbe9f4; }

.title { 
    font-size: 38px; 
    font-weight: 800; 
    color: #1f3c88; 
    text-align: center; 
}

.subtitle { 
    font-size: 17px; 
    color: #444; 
    text-align: center; 
    margin-bottom: 25px; 
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

#Prediction 
if submitted:
    try:
        result = predict_credit_risk(
            age, sex, job, housing,
            saving_accounts, checking_account,
            credit_amount, duration
        )

        if result.lower() == "good":
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
