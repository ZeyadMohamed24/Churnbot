import pandas as pd
import json
import numpy as np
import re


def clean_numeric_value(text):
    NUMERIC_PATTERN = re.compile(r"[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?")
    match = NUMERIC_PATTERN.search(text)
    return float(match.group().replace(",", "")) if match else "none"


def parse_json_string(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None


def replace_none_in_column(
    df: pd.DataFrame,
    column: str,
    mean_values: pd.Series,
    mode_values: pd.Series,
    categorical_cols: list,
) -> pd.DataFrame:
    if column in categorical_cols:
        # Replace 'none' with mode for categorical columns
        df[column] = df[column].replace("none", mode_values[column])
    else:
        # Replace 'none' with NaN and then fill with mean for numerical columns
        df[column] = df[column].replace("none", np.nan)
        df[column] = df[column].astype(float)
        df[column] = df[column].fillna(mean_values[column])
    return df
