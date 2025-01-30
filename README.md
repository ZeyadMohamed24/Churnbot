# Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques.

## Folder Structure

- **data/**: Stores all data files.
- **docs/**: Contains project documentation files.
- **models/**: Trained model files.
- **notebooks/**: Jupyter notebooks for EDA and experimentation.
- **reports/**: Generated reports and visualizations.
- **src/**: Source code for data processing, feature engineering, model training, and visualization.
- **venv/**: Virtual Environment files.

## Setup Instructions

1. Create a virtual environment:
    ```bash
    python -m venv venv
    source .\venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install setup:
    ```bash
    pip install -e .
    ```

4. Run the main program:
    ```bash
    run_main
    ```

5. Start Ollama:
    ```bash
    ollama serve
    ```  
6. Run the Fast API:
    ```bash
    uvicorn src.api.app:app --reload
    ```  

7. Launch the streamlit application:
    ```bash
    streamlit run src/app/app.py
    ```    

## If you want to check the api
Open swagger from this link:
    ```
    http://127.0.0.1:8000/docs
    ```    
## You can also run a single program

Run data cleaning script:
    ```
    clean_data
    ```

Engineer Features:
    ```
    engineer_features
    ```

Train the model:
    ```
    train_model
    ```

Predict churn from testing data:
    ```
    predict
    ```    

<p style="color:red;font-size:20px">If an internal server error occurs try: </p>Ollama run llama3  