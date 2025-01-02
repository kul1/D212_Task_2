# config_utils.py

# Base directory for all result outputs
RESULTS_BASE_DIR = "results"

# Function to get the appropriate results directory based on the config type
def get_results_dir(config_type):
    return f"{RESULTS_BASE_DIR}/{config_type}/"
