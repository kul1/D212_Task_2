# File Path: utils/data_loading.py

import os
import pandas as pd

def load_raw_data(config, log_debug):
    """
    Loads the raw data from the specified file path in the configuration.

    Parameters:
    - config (module): The configuration module containing paths and file names.
    - log_debug (function): The function to log debug messages.

    Returns:
    - pd.DataFrame: The loaded DataFrame containing the raw data.
    """
    # Get the file path from the configuration
    raw_data_file = os.path.join(config.RAW_DATA_DIR, config.RAW_DATA_FILE)

    # Load the raw data
    data = pd.read_csv(raw_data_file)
    print(f"\n### Loading Raw Data from {config.RAW_DATA_FILE} ###")

    # Log debug information
    log_debug(f"Raw data loaded from {config.RAW_DATA_FILE}")
    log_debug(data.head())

    return data

def load_prepared_data(config, log_debug):
    """
    Loads the prepared data file with dummy variables from the specified file path in the configuration.

    Parameters:
    - config (module): The configuration module containing paths and file names.
    - log_debug (function): The function to log debug messages.

    Returns:
    - pd.DataFrame: The loaded DataFrame containing the prepared data with dummy variables.
    """
    # Get the file path from the configuration
    prepared_data_file = os.path.join(config.PREPARED_DATA_DIR, config.PREPARED_DATA_FILE)

    # Load the prepared data
    data_with_dummies = pd.read_csv(prepared_data_file)
    print(f"\n### Data loaded from {prepared_data_file} ###")

    # Log debug information
    log_debug(f"Columns: {data_with_dummies.columns.tolist()}")
    log_debug(f"\n### Data After Loading ###\n{data_with_dummies.head()}")

    return data_with_dummies

