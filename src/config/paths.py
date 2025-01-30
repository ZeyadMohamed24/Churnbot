import os
from pathlib import Path

# Base
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Models
MODELS_DIR = BASE_DIR / "models"
CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
ONE_HOT_ENCODER_PATH = os.path.join(MODELS_DIR, "one_hot_encoder.joblib")
STANDARD_SCALER_PATH = os.path.join(MODELS_DIR, "standard_scaler.joblib")

# Data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw/customer_churn_dataset-training-master.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "processed/clean_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed/processed_data.csv")

# Reports
REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_PATH = os.path.join(REPORTS_DIR, "predictions.csv")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.csv")
PLOTS_PATH = os.path.join(REPORTS_DIR, "plots")
