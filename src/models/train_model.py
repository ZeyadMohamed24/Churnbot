import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from ..config.paths import PROCESSED_DATA_PATH, CHURN_MODEL_PATH


def train_model(data_filepath: str, model_filepath: str) -> None:
    """
    Train the model on processed data.

    Parameters:
    - data_filepath: the path to the processed data.
    - model_filepath: the path to the model saving location.

    Returns: None.
    """
    data = pd.read_csv(data_filepath)
    features = data.drop(columns="Churn")
    target = data["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=3, alpha=1.0)
    model.fit(x_train, y_train)

    joblib.dump(model, model_filepath)


def main():
    train_model(
        PROCESSED_DATA_PATH,
        CHURN_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
