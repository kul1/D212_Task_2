# File: flow/flow_data_mining.py

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils.generate_summary_statistics import generate_summary_statistics
from utils.config_loader import load_config  # Load configuration dynamically

# File: flow/flow_data_mining.py
# File: flow/flow_data_mining.py
# File: flow/flow_data_mining.py

import pandas as pd
from utils.generate_summary_statistics import generate_summary_statistics

def run_analysis(data_with_dummies, config):
    # Get the target variable
    target_var = config.TARGET_COLUMN

    # Check if the target variable exists in the DataFrame
    if target_var not in data_with_dummies.columns:
        raise ValueError(f"Target variable '{target_var}' not found in the DataFrame.")

    # Identify predictor variables
    predictor_vars = [col for col in data_with_dummies.columns if col != target_var]

    # Check if there are any predictor variables
    if not predictor_vars:
        raise ValueError("No predictor variables found. Please check your dataset.")

    # Log research question and method justification
    print("Research Question: What factors influence High Blood Pressure?")
    print("Method Chosen: k-nearest neighbor (KNN) for classification.")

    # Check for initial NaNs in the target variable and remove them if present
    if data_with_dummies[target_var].isna().sum() > 0:
        data_with_dummies = data_with_dummies.dropna(subset=[target_var])

    # Check for an empty DataFrame after NaN removal
    if data_with_dummies.empty:
        raise ValueError("DataFrame is empty after dropping rows with NaNs in the target column. Cannot proceed.")

    # Generate summary statistics for continuous and transformed variables
    generate_summary_statistics(data_with_dummies, target_var, config.CONTINUOUS_COLUMNS, config.TRANSFORMED_COLUMNS)

    # Distinguish between classification and regression
    model = None  # Initialize model variable

    if config.CONFIG_TYPE == 'knn':
        print("\n### Running KNN Classification Analysis ###")
        from flow.knn_analysis import run_knn_classification  # Import KNN analysis module
        model = run_knn_classification(data_with_dummies, config, predictor_vars, target_var)  # Capture model

    elif config.CONFIG_TYPE == 'naive_bayes':
        print("\n### Running Naive Bayes Classification Analysis ###")
        from flow.naive_bayes_analysis import run_naive_bayes_classification  # Import Naive Bayes analysis module
        model = run_naive_bayes_classification(data_with_dummies, config, predictor_vars, target_var)  # Capture model

    elif config.CONFIG_TYPE in ['linear', 'logistic']:
        print("\n### Running Regression Analysis ###")
        from flow.regression_analysis import run_regression  # Import Regression analysis module
        model = run_regression(data_with_dummies, config, predictor_vars, target_var)  # Capture model

    else:
        print("Invalid CONFIG_TYPE specified.")

    return model, predictor_vars  # Return model and predictor variables

