# data_cleaning.py

import pandas as pd
import numpy as np
from utils.config_loader import load_config  # Use the unified config loading method

# Load the configuration module
config = load_config()

# Use the loaded config to get continuous and categorical column definitions
CONTINUOUS_COLUMNS = config.CONTINUOUS_COLUMNS
CATEGORICAL_COLUMNS = config.CATEGORICAL_COLUMNS

def clean_data(data):
    """
    Clean the dataset by handling missing values and outliers, and imputing values.

    Parameters:
    - data (pd.DataFrame): The dataset to clean.

    Returns:
    - pd.DataFrame: The cleaned dataset.
    """
    # Fill missing values for continuous columns with the median
    for col in CONTINUOUS_COLUMNS:
        data[col] = data[col].fillna(data[col].median())

    print("Missing values after numerical imputation:")
    print(data[CONTINUOUS_COLUMNS].isnull().sum())

    # Fill missing values for categorical columns with the mode
    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].fillna(data[col].mode().iloc[0])

    print("Missing values after categorical imputation:")
    print(data[CATEGORICAL_COLUMNS].isnull().sum())

    # Handle outliers by capping them within the 1.5*IQR range
    for col in CONTINUOUS_COLUMNS:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.where((data[col] < lower_bound) | (data[col] > upper_bound), np.nan, data[col])

    print("Missing values after outlier handling:")
    print(data[CONTINUOUS_COLUMNS].isnull().sum())

    # Re-impute missing values caused by outlier handling
    for col in CONTINUOUS_COLUMNS:
        data[col] = data[col].fillna(data[col].median())

    print("Missing values after re-imputation:")
    print(data[CONTINUOUS_COLUMNS].isnull().sum())

    return data
