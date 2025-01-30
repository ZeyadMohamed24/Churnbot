import pandas as pd
from .data.data_cleaning import clean_data
from .features.feature_engineering import engineer_features
from .models.train_model import train_model
from .scripts.predict import predict
from .config.paths import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    PROCESSED_DATA_PATH,
    CHURN_MODEL_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH,
    PLOTS_PATH,
)


def main():
    # Data Cleaning
    clean_data(
        RAW_DATA_PATH,
        CLEAN_DATA_PATH,
    )
    print("Data is Cleaned")

    # Feature Engineering
    data = pd.read_csv(CLEAN_DATA_PATH)
    processed_data = engineer_features(data)
    processed_data.to_csv(
        PROCESSED_DATA_PATH,
        index=None,
    )
    print("Features are engineered")

    # Model Training
    train_model(
        PROCESSED_DATA_PATH,
        CHURN_MODEL_PATH,
    )
    print("Model Trained")

    # Model Prediction
    predict(
        PROCESSED_DATA_PATH,
        CHURN_MODEL_PATH,
        PREDICTIONS_PATH,
        METRICS_PATH,
        PLOTS_PATH,
    )
    print("Execution Completed !")


if __name__ == "__main__":
    main()
