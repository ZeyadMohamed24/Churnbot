from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import ollama
from .feature_extraction_from_text import extract_specific_features
from .utils import replace_none_in_column
from ..config.paths import (
    LABEL_ENCODER_PATH,
    ONE_HOT_ENCODER_PATH,
    CHURN_MODEL_PATH,
    RAW_DATA_PATH,
    STANDARD_SCALER_PATH,
)

app = FastAPI()


class TextInput(BaseModel):
    text: str


# Load models and encoders
standard_scaler = load(STANDARD_SCALER_PATH)
label_encoder = load(LABEL_ENCODER_PATH)
one_hot_encoder = load(ONE_HOT_ENCODER_PATH)
model = load(CHURN_MODEL_PATH)
data = pd.read_csv(RAW_DATA_PATH)

# Define columns
numerical_cols = data.select_dtypes(include=["number"]).columns
categorical_cols = data.select_dtypes(include=["object"]).columns
mean_values = data[numerical_cols].mean()
mode_values = data[categorical_cols].mode().iloc[0]


def make_prediction(user_text: str) -> int:
    features = extract_specific_features(user_text)
    features_df = pd.DataFrame([features])
    print(f"text features :{features_df}")

    required_columns = [
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction",
    ]
    for col in required_columns:
        if col not in features_df.columns:
            features_df[col] = "None"

    for col in features_df.columns:
        features_df = replace_none_in_column(
            features_df, col, mean_values, mode_values, categorical_cols
        )
    print(f"into model features :{features_df}")

    contract_length_encoded = one_hot_encoder.transform(
        features_df[["Subscription Type", "Contract Length"]]
    )
    contract_length_encoded_df = pd.DataFrame(
        contract_length_encoded,
        columns=one_hot_encoder.get_feature_names_out(
            ["Subscription Type", "Contract Length"]
        ),
    )

    features_df["Gender"] = label_encoder.transform(features_df["Gender"])

    features_df = pd.concat(
        [
            features_df.drop(columns=["Subscription Type", "Contract Length"]),
            contract_length_encoded_df,
        ],
        axis=1,
    )
    scaled_df = standard_scaler.transform(features_df)
    prediction = model.predict(scaled_df)
    print(f"Prediction : {prediction[0]}")  
    print(f"features used for prediction :{scaled_df[0]}") 
    return int(prediction[0])


@app.post("/chat")
async def chat(input: TextInput):
    user_input = input.text.lower()

    if "prediction" in user_input or "predict" in user_input:
        prediction_result = make_prediction(user_input)
        if prediction_result == 1:
            prediction_text = "will churn"
        else:
            prediction_text = "will not churn"
        response_content = f"Based on model prediction this user {prediction_text}. How can I assist you further?"
    else:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
        )
        response_content = response["message"]["content"]

    return {"response": response_content}
