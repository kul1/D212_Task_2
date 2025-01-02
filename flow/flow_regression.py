# flow/flow_regression.py

import pandas as pd
import statsmodels.api as sm
from utils.vif_calculation import calculate_vif
from utils.save_results import save_vif_results
from utils.model_comparison import compare_models
from utils.visualizations import create_visualizations  # Ensure this handles visualizations
from utils.generate_summary_statistics import generate_summary_statistics
from utils.residual_analysis import calculate_rse, plot_residuals
from utils.variable_selection import display_variable_selection_table
from utils.config_loader import load_config  # Load configuration dynamically


def run_analysis(data_with_dummies, config):
    target_var = config.TARGET_COLUMN

    # Step 6.1: Check for NaNs in the target variable and remove them if present
    if data_with_dummies[target_var].isna().sum() > 0:
        data_with_dummies = data_with_dummies.dropna(subset=[target_var])

    # Check for an empty DataFrame after NaN removal
    if data_with_dummies.empty:
        raise ValueError("DataFrame is empty after dropping rows with NaNs in the target column. Cannot proceed with regression.")

    # Step 7: Define predictor variables from transformed columns (excluding target)
    predictor_vars = [col for col in data_with_dummies.columns if col != target_var]

    # Output details about regression setup for reporting
    print(f"REGRESSION TYPE: {config.CONFIG_TYPE}")
    print(f"TARGET_COLUMN: {config.TARGET_COLUMN}")

    # Step 8: Generate summary statistics for continuous and transformed variables
    generate_summary_statistics(data_with_dummies, target_var, config.CONTINUOUS_COLUMNS, config.TRANSFORMED_COLUMNS)

    # Step 9: Initial regression using all predictors
    X_initial = sm.add_constant(data_with_dummies[predictor_vars])
    y = data_with_dummies[target_var]
    initial_model = sm.OLS(y, X_initial).fit()

    print("\n### Initial Regression Results ###\n")
    print(initial_model.summary())

    # Step 10: Run VIF to filter out high VIF variables
    vif_filtered_vars, vif_data = calculate_vif(data_with_dummies, config)
    save_vif_results(vif_data)

    # Step 11: Final Check for NaNs Before Running Reduction Methods
    if data_with_dummies[vif_filtered_vars].isna().sum().sum() > 0:
        raise ValueError("NaN values found in predictor variables. Reduction methods cannot proceed.")

    # Additional steps for VIF and variable selection can be added here...

    # Generate final visualizations
    create_visualizations(data_with_dummies, target_var, predictor_vars, config, step='final_model')


