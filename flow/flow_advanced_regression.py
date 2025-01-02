import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.linear_model import Lasso
from utils.vif_calculation import calculate_vif
from utils.save_results import save_vif_results
from utils.visualizations import create_visualizations
from utils.generate_summary_statistics import generate_summary_statistics

def run_analysis(data_with_dummies, config):
    target_var = config.TARGET_COLUMN
    # Exclude ReAdmis_Yes from predictors
    predictor_vars = [var for var in config.TRANSFORMED_COLUMNS if var != 'ReAdmis_Yes']

    # Check for initial NaNs in the target column
    initial_nan_count = data_with_dummies[target_var].isna().sum()
    unique_values = data_with_dummies[target_var].unique()
    print(f"Initial NaNs in target column '{target_var}': {initial_nan_count}")
    print(f"Unique values in target column '{target_var}': {unique_values}")

    # Check for NaNs in the target variable and remove them if present
    if initial_nan_count > 0:
        data_with_dummies = data_with_dummies.dropna(subset=[target_var])

    # Check for an empty DataFrame after NaN removal
    if data_with_dummies.empty:
        raise ValueError("DataFrame is empty after dropping rows with NaNs in the target column. Cannot proceed with regression.")

    # Output details about regression setup for reporting
    print(f"REGRESSION TYPE: {config.CONFIG_TYPE}")
    print(f"TARGET_COLUMN: {config.TARGET_COLUMN}")
    print(f"REGRESSION_METHODS: {', '.join(config.REGRESSION_METHODS)}")
    print(f"VIF_THRESHOLD: {config.VIF_THRESHOLD}")
    print(f"COLUMN TYPE: {config.COLUMN_CONFIG}")
    print(f"COLUMN TRANSFORM: {predictor_vars}")

    # Generate summary statistics for continuous and transformed variables
    generate_summary_statistics(data_with_dummies, target_var, config.CONTINUOUS_COLUMNS, predictor_vars)

    # Run VIF to filter out high VIF variables
    vif_filtered_vars, vif_data = calculate_vif(data_with_dummies[predictor_vars], config)
    save_vif_results(vif_data)

    # Final Check for NaNs Before Running Lasso Regression
    if data_with_dummies[vif_filtered_vars].isna().sum().sum() > 0:
        raise ValueError("NaN values found in predictor variables. Lasso regression cannot proceed.")

    # Lasso regression using filtered predictors
    X_filtered = data_with_dummies[vif_filtered_vars]  # Use only the filtered variables
    y = data_with_dummies[target_var]

    lasso_model = Lasso(alpha=1.0)  # Adjust alpha as needed for regularization
    lasso_model.fit(X_filtered, y)

    # Predictions
    y_pred = lasso_model.predict(X_filtered).round()  # Round for binary classification
    mse = mean_squared_error(y, y_pred)

    print("\n### Lasso Regression Results ###\n")
    print(f"Mean Squared Error: {mse}")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    print("Confusion Matrix:\n", conf_matrix)  # Debugging line

    # Calculate accuracy
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
    print(f"Model Accuracy: {accuracy:.4f}")

    # Create a classification report
    class_report = classification_report(y, y_pred, output_dict=True)

    # Generate final visualizations
    create_visualizations(data_with_dummies, target_var, predictor_vars, conf_matrix, class_report, config, step='final_model')
