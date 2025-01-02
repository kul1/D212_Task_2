# utils/check_numeric_and_clean.py

import numpy as np
import pandas as pd

def check_numeric_and_clean(data_with_dummies, predictor_vars, target_var):
    """
    Ensure that all columns in the dataset are numeric and clean any remaining NaN or invalid values.

    Parameters:
    - data_with_dummies (pd.DataFrame): The dataset with dummy variables.
    - predictor_vars (list): List of predictor variables.
    - target_var (str): The target variable.

    Returns:
    - pd.DataFrame: The cleaned dataset with numeric values.
    """

    # Step 1: Convert any boolean columns to integers
    print("\n### Converting Boolean Columns to Integers ###")
    for col in data_with_dummies.select_dtypes(include=['bool']).columns:
        data_with_dummies[col] = data_with_dummies[col].astype(int)

    # Step 2: Ensure all columns are numeric
    print("\n### Ensuring All Columns Are Numeric ###")
    data_with_dummies = data_with_dummies.apply(pd.to_numeric, errors='coerce')

    # Step 3: Check for NaNs or invalid values in predictor and target columns
    print("\n### Checking for NaNs or Invalid Values in Predictor and Target Columns ###")
    print("NaNs in predictor columns:\n", data_with_dummies[predictor_vars].isna().sum())
    print("NaNs in target column:\n", data_with_dummies[target_var].isna().sum())

    # Step 4: Drop any rows with NaN values in either the predictor or target columns
    print("\n### Dropping Rows with NaN Values ###")
    data_with_dummies.dropna(subset=predictor_vars + [target_var], inplace=True)

    # Step 5: Final check that all data is numeric
    print("\n### Checking Data Types Before Running Regression ###")
    print("Predictor variables types:\n", data_with_dummies[predictor_vars].dtypes)
    print("Target variable type:\n", data_with_dummies[target_var].dtype)

    # If any non-numeric columns are found, raise an error
    if not np.issubdtype(data_with_dummies[predictor_vars].dtypes.values[0], np.number):
        raise ValueError("Predictor variables contain non-numeric values.")
    if not np.issubdtype(data_with_dummies[target_var].dtype, np.number):
        raise ValueError("Target variable contains non-numeric values.")

    return data_with_dummies
