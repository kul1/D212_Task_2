CONFIG_TYPE = 'pca_analysis'

# File paths
RAW_DATA_DIR = 'rawdata'
RAW_DATA_FILE = 'medical_clean.csv'
PREPARED_DATA_DIR = f'prepared/{CONFIG_TYPE}'
PREPARED_DATA_FILE = f'prepared_data_for_{CONFIG_TYPE}.csv'
RESULTS_DIR = f'results/{CONFIG_TYPE}'
VISUALS_DIR = f'{RESULTS_DIR}/visuals'

# Flow function
FLOW_FUNCTION = f'flow.flow_{CONFIG_TYPE}.run_pca_analysis'

# Analysis parameters
PCA_COMPONENTS_RETAINED = 5
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Column configurations
CONTINUOUS_COLUMNS = [
    'Age', 'Income', 'VitD_levels', 'Doc_visits', 'TotalCharge'
]
EXCLUDED_COLUMNS = ['ID', 'Customer_id']

# Define column types
COLUMN_CONFIG = {
    'Age': 'continuous',
    'Income': 'continuous',
    'VitD_levels': 'continuous',
    'Doc_visits': 'continuous',
    'Overweight': 'categorical',  # Example target
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

# Define continuous and categorical columns
CONTINUOUS_COLUMNS = ['Age', 'Income', 'VitD_levels', 'Doc_visits']
CATEGORICAL_COLUMNS = [col for col, col_type in COLUMN_CONFIG.items() if col_type == 'categorical']

# Specify the target column
TARGET_COLUMN = 'Overweight_Yes'

# Define independent variables (update to include dummy variables)
INDEPENDENT_VARIABLES = CONTINUOUS_COLUMNS + [
    'Stroke_Yes', 'Arthritis_Yes', 'Diabetes_Yes', 'Hyperlipidemia_Yes',
    'BackPain_Yes', 'Anxiety_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes', 'Asthma_Yes'
]

# CLASS_LABELS for confusion matrix labeling
CLASS_LABELS = ['Not Overweight', 'Overweight']  # Adjust labels based on data

# KNN Configuration
KNN_CONFIG = {
    "n_neighbors": 5,          # Default to 5 neighbors
    "weights": "uniform",      # Weight function used in prediction
    "algorithm": "auto",       # Algorithm used to compute nearest neighbors
    "metric": "minkowski",     # Distance metric
}
# Other configurations
TEST_SIZE = 0.2
RANDOM_STATE = 42
