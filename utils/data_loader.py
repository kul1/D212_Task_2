# utils/data_loader.py

import os
import pandas as pd
from utils.config_loader import load_config

config = load_config()

def load_data(save_dir, save_file):
    """
    Load the dataset from the prepared data file specified in the config.

    Parameters:
    - save_dir (str): The directory to load the data from.
    - save_file (str): The file name to load the data from.

    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    prepared_data_path = os.path.join(save_dir, save_file)

    if not os.path.exists(prepared_data_path):
        raise FileNotFoundError(f"Prepared data not found: {prepared_data_path}")

    return pd.read_csv(prepared_data_path)

def save_prepared_data_with_dummies(data_with_dummies, save_dir, save_file):
    """
    Save the transformed data (with dummy variables) to the specified prepared data file.

    Parameters:
    - data_with_dummies (pd.DataFrame): The data with dummy variables.
    - save_dir (str): The directory to save the data.
    - save_file (str): The file name to save the data.

    Returns:
    - None
    """
    save_path = os.path.join(save_dir, save_file)
    data_with_dummies.to_csv(save_path, index=False)
    print(f"Data with dummies saved to {save_path}")
