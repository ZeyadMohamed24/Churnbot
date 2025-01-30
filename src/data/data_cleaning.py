import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
from ..config.paths import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    LABEL_ENCODER_PATH,
    ONE_HOT_ENCODER_PATH,
)


def clean_data(input_filepath: str, output_filepath: str) -> None:
    """
    Clean Dataset from nulls, duplicates, redundant columns and encode the data.

    Parameters:
    - input_filepath: the path to the dataset.
    - output_filepath: the path to the cleaned dataset saving location.

    Returns: None
    """
    raw_data = pd.read_csv(input_filepath)
    clean_data = raw_data.dropna()

    encoder = LabelEncoder()
    clean_data["Gender"] = encoder.fit_transform(clean_data["Gender"])

    categorical_columns = [
        col for col in clean_data.columns if clean_data[col].dtype == "object"
    ]
    one_hot_encoder = OneHotEncoder(sparse_output=False)

    if categorical_columns:
        encoded_features = one_hot_encoder.fit_transform(
            clean_data[categorical_columns]
        )
        encoded_feature_names = one_hot_encoder.get_feature_names_out(
            categorical_columns
        )
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
        clean_data = pd.concat(
            [clean_data.reset_index(drop=True), encoded_df.reset_index(drop=True)],
            axis=1,
        )
        clean_data.drop(columns=categorical_columns, inplace=True)

    if "CustomerID" in clean_data.columns:
        clean_data.drop(columns="CustomerID", inplace=True)

    joblib.dump(encoder, LABEL_ENCODER_PATH)
    joblib.dump(
        one_hot_encoder,
        ONE_HOT_ENCODER_PATH,
    )

    clean_data.to_csv(output_filepath, index=None)


def main():
    clean_data(
        RAW_DATA_PATH,
        CLEAN_DATA_PATH,
    )


if __name__ == "__main__":
    main()
