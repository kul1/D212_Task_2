# File Path: utils/data_cleaning.py

import pandas as pd
from utils.config_loader import load_config  # Import the function to load the config

# Load the configuration module
config = load_config()  # Ensure this loads the correct configuration based on your requirements

def clean_and_create_dummies(data, config):
    """Cleans the data and creates dummy variables based on configuration."""
    try:
        # Drop duplicates
        data = data.drop_duplicates()

        # Handle missing values based on the configuration
        for column in config.CATEGORICAL_COLUMNS:
            data[column].fillna('Unknown', inplace=True)

        for column in config.CONTINUOUS_COLUMNS:
            data[column].fillna(data[column].mean(), inplace=True)

        # Create dummy variables for categorical columns
        data_with_dummies = pd.get_dummies(data, columns=config.CATEGORICAL_COLUMNS, drop_first=True)

        return data_with_dummies

    except Exception as e:
        print(f"Error during data cleaning and dummy variable creation: {e}")
        return None

def clean_data_and_create_dummies(data, config, log_debug):
    """Cleans the data and creates dummy variables based on configuration."""
    log_debug("Starting data cleaning and dummy creation process.")

    # Clean the data and create dummy variables
    data_with_dummies = clean_and_create_dummies(data, config)

    if data_with_dummies is not None:
        log_debug(f"\n### Data After Cleaning and Creating Dummies ###\n{data_with_dummies.head()}")
    else:
        log_debug("Data cleaning and dummy creation failed. Please check the logs for errors.")

    return data_with_dummies

