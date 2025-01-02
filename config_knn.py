from config_utils import get_results_dir  # Import the utility to set the results directory

# ----------------------------
# Configuration Settings
# ----------------------------

# Analysis type for this config (important for dynamically handling save paths)
CONFIG_TYPE = 'knn'  # Set to 'knn' for K-nearest neighbors analysis

# KNN-specific parameters
TARGET_COLUMN = 'HighBlood'  # Change target variable for KNN classification

# Target variable mapping for classification (if applicable)
TARGET_VARIABLE_MAPPING = {'Yes': 1, 'No': 0}  # Mapping for the target variable

# Define COLUMN_CONFIG similar to previous setup
COLUMN_CONFIG = {
    'Age': 'continuous',
    'Income': 'continuous',
    'Overweight': 'categorical',
    'Soft_drink': 'categorical',
    'Stroke': 'categorical',
    'Arthritis': 'categorical',
    'Diabetes': 'categorical',
    'Hyperlipidemia': 'categorical',
    'BackPain': 'categorical',
    'Anxiety': 'categorical',
    'Allergic_rhinitis': 'categorical',
    'Reflux_esophagitis': 'categorical',
    'Asthma': 'categorical'
}

# Dynamically generate continuous and categorical columns based on the COLUMN_CONFIG
CONTINUOUS_COLUMNS = [col for col, col_type in COLUMN_CONFIG.items() if col_type == 'continuous']
CATEGORICAL_COLUMNS = [col for col, col_type in COLUMN_CONFIG.items() if col_type == 'categorical']

# Update INDEPENDENT_VARIABLES to reflect the columns after dummy encoding
INDEPENDENT_VARIABLES = [
    'Age',
    'Income',
    'Overweight_Yes',  # Updated to the dummy variables
    'Soft_drink_Yes',  # Updated to the dummy variables
    'Stroke_Yes',      # Updated to the dummy variables
    'Arthritis_Yes',   # Updated to the dummy variables
    'Diabetes_Yes',    # Updated to the dummy variables
    'Hyperlipidemia_Yes',  # Updated to the dummy variables
    'BackPain_Yes',    # Updated to the dummy variables
    'Anxiety_Yes',     # Updated to the dummy variables
    'Allergic_rhinitis_Yes',  # Updated to the dummy variables
    'Reflux_esophagitis_Yes',  # Updated to the dummy variables
    'Asthma_Yes'       # Updated to the dummy variables
]

# KNN configuration settings
KNN_CONFIG = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
    'metric': 'euclidean'
}

# Directory and file paths for raw and prepared data
RAW_DATA_DIR = 'rawdata'
RAW_DATA_FILE = 'medical_clean.csv'
PREPARED_DATA_DIR = 'prepared/knn'
PREPARED_DATA_FILE = 'prepared_data_with_dummies.csv'

# Automatically set the results directory based on CONFIG_TYPE
RESULTS_DIR = get_results_dir(CONFIG_TYPE)

# Specify the visualization directory
VISUALIZATION_DIR = RESULTS_DIR + 'visuals/'  # Ensure it points to the right path

# Specify the analysis function for KNN
ANALYSIS_FUNCTION = 'flow.knn_analysis.run_knn_classification'

# New attributes for dynamic test size and random state
TEST_SIZE = 0.2  # Proportion of the dataset to include in the test split
RANDOM_STATE = 42  # Seed used by the random number generator

# Add CLASS_LABELS for confusion matrix labeling
CLASS_LABELS = ['0', '1']  # Change to the appropriate labels for your classification
