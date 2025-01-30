import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from ..visualization.visuals import (
    save_confusion_matrix,
    save_roc_curve,
    save_histogram,
    save_density_plot,
    save_feature_importance,
)
from ..config.paths import (
    PROCESSED_DATA_PATH,
    CHURN_MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    PLOTS_PATH,
)


def predict(
    input_filepath: str,
    model_filepath: str,
    output_filepath: str,
    metrics_filepath: str,
    plots_dir: str,
) -> None:
    """
    Predict churn from testing data and evaluate the model.

    Parameters:
    - input_filepath: Path to the test data.
    - model_filepath: Path to the model file.
    - output_filepath: Path to save the predictions.
    - metrics_filepath: Path to save the evaluation metrics.
    - plots_dir: Directory path to save the plots.

    Returns: None
    """
    data = pd.read_csv(input_filepath)
    features = data.drop(columns="Churn")
    true_labels = data["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, true_labels, test_size=0.2, random_state=42
    )
    features = data.drop(columns="Churn", axis=1)
    true_labels = data["Churn"]

    model = joblib.load(model_filepath)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    results = pd.DataFrame({"Prediction": predictions})
    results.to_csv(output_filepath, index=False)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC Score": roc_auc,
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    metrics_df.to_csv(metrics_filepath, index=False)

    save_confusion_matrix(y_test, predictions, plots_dir)
    save_roc_curve(y_test, probabilities, roc_auc, plots_dir)
    save_histogram(y_test, predictions, plots_dir)
    save_density_plot(y_test, predictions, plots_dir)
    save_feature_importance(model, features.columns, plots_dir)


def main():
    predict(
        PROCESSED_DATA_PATH,
        CHURN_MODEL_PATH,
        PREDICTIONS_PATH,
        METRICS_PATH,
        PLOTS_PATH,
    )


if __name__ == "__main__":
    main()
