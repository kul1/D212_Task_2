import os
import pandas as pd
from utils.config_loader import load_config, import_flow_function
from utils.directory_setup import setup_directories
from utils.clean_and_create_data import clean_and_create_data

def main():
    try:
        # Step 1: Load configuration
        config = load_config()
        print(f"Loaded configuration for {config.CONFIG_TYPE}.")

        # Step 2: Setup directories
        setup_directories(config)

        # Step 3: Load raw data
        data_path = os.path.join(config.RAW_DATA_DIR, config.RAW_DATA_FILE)
        data = pd.read_csv(data_path)
        print(f"Data loaded from {data_path} with shape {data.shape}")

        # Step 4: Clean and prepare data
        data_cleaned = clean_and_create_data(data, config)
        prepared_data_file = os.path.join(config.PREPARED_DATA_DIR, config.PREPARED_DATA_FILE)
        data_cleaned.to_csv(prepared_data_file, index=False)
        print(f"Prepared data saved to {prepared_data_file}")

        # Step 5: Execute analysis
        flow_function = import_flow_function(config.FLOW_FUNCTION)
        flow_function(data_cleaned, config)

    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
