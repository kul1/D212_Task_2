# data_encoding.py
import pandas as pd

def create_dummies(data, categorical_columns, target_column):
    """
    Create dummy variables for the specified categorical columns, ensuring the target column is not included.

    Parameters:
    - data (pd.DataFrame): The dataset to encode.
    - categorical_columns (list of str): The columns to convert into dummy variables.
    - target_column (str): The target column which should not be converted into dummies.

    Returns:
    - pd.DataFrame: The dataset with categorical columns encoded as dummy variables.
    """
    # Exclude the target column from being dummified
    categorical_columns = [col for col in categorical_columns if col != target_column]

    # Generate dummies for categorical columns
    data_with_dummies = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    return data_with_dummies
