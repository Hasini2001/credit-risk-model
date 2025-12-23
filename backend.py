import pandas as pd
import joblib

# Load model & encoders only once
model = joblib.load("best_credit_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")


def predict_credit_risk(
    age, sex, job, housing,
    saving_accounts, checking_account,
    credit_amount, duration
):
    # Create raw input dataframe
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

    # One-hot encoding
    input_encoded = pd.get_dummies(input_raw, drop_first=True)

    # Align with training columns
    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict
    pred = model.predict(input_encoded)[0]
    pred_label = target_encoder.inverse_transform([pred])[0]

    return pred_label
