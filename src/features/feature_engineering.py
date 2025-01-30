import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from ..config.paths import STANDARD_SCALER_PATH, CLEAN_DATA_PATH, PROCESSED_DATA_PATH


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features and create new features if needed.

    Parameters:
    - data: the dataframe which contains the cleaned dataset.

    Returns:
    - processed_data.
    """
    features = data.drop(columns="Churn")
    target = data["Churn"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    scaled_df = pd.concat([scaled_features_df, target.reset_index(drop=True)], axis=1)

    joblib.dump(scaler, STANDARD_SCALER_PATH)

    return scaled_df


def main():
    data = pd.read_csv(CLEAN_DATA_PATH)
    processed_data = engineer_features(data)
    processed_data.to_csv(
        PROCESSED_DATA_PATH,
        index=None,
    )


if __name__ == "__main__":
    main()
