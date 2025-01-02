import os
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_encoding import create_dummies
from utils.data_loader import save_prepared_data_with_dummies
from data_cleaning import clean_data
from sklearn.model_selection import train_test_split
import pandas as pd

def clean_and_create_dummies(data, config):
    """
    Clean the raw data including the target column and create dummy variables.

    Parameters:
    - data (pd.DataFrame): The raw dataset.
    - config: The configuration object containing column information.

    Returns:
    - pd.DataFrame: The cleaned dataset with dummy variables created.
    """
    # Step 1: Handle missing values for numerical columns
    numerical_columns = [col for col, col_type in config.COLUMN_CONFIG.items() if col_type == 'continuous']
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Step 2: Handle missing values for categorical columns
    categorical_columns = [col for col, col_type in config.COLUMN_CONFIG.items() if col_type == 'categorical']
    for col in categorical_columns:
        data[col] = data[col].fillna('Unknown')  # Replace NaN with 'Unknown'

    # Step 3: Create dummy variables for categorical columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Step 4: Convert boolean columns to integers (0 and 1)
    boolean_columns = [col for col in data.columns if data[col].dtype == bool]
    for col in boolean_columns:
        data[col] = data[col].astype(int)

    # Validate cleaning
    assert not data.isnull().any().any(), "Data cleaning failed. NaN values remain."

    return data
