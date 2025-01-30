from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="0.4",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "joblib",
        "xgboost",
        "fastapi",
        "pydantic",
        "ollama",
        "gliner",
        "torch",
        "streamlit",
    ],
    entry_points={
        "console_scripts": [
            "run_main=src.main:main",
            "clean_data=src.data.data_cleaning:main",
            "engineer_features=src.features.feature_engineering:main",
            "train_model=src.models.train_model:main",
            "predict=src.scripts.predict:main",
            "predict_from_text=src.pipeline.app:main",
        ],
    },
)
