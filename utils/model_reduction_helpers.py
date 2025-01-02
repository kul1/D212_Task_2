# File: utils/model_reduction_helpers.py

from utils.save_results import save_model_metrics, save_variable_selection
from reduction_methods.run_reduction_method import run_reduction_method
import pandas as pd
import statsmodels.api as sm

def run_reduction_methods(data_with_dummies, vif_filtered_vars, target_var, config):
    """
    Run the regression reduction methods dynamically from the config file.

    Parameters:
    - data_with_dummies (pd.DataFrame): DataFrame containing the cleaned data with dummy variables.
    - vif_filtered_vars (list): The variables filtered by VIF.
    - target_var (str): The target variable for the regression.
    - config (module): Configuration file containing settings for the reduction methods.

    Returns:
    - reduced_results_dict (dict): A dictionary of reduced models and their selected variables.
    """
    reduced_results_dict = {}

    for method in config.REGRESSION_METHODS:
        print(f"\n### Running {method.capitalize()} Reduction ###")
        # Run the specified reduction method dynamically
        reduced_model, selected_variables = run_reduction_method(
            method, data_with_dummies, target_var, vif_filtered_vars, config
        )
        reduced_results_dict[method] = (reduced_model, selected_variables)

    return reduced_results_dict

def extract_selected_variables(data_with_dummies, initial_vars, vif_vars, reduced_results_dict, config):
    """
    Extract the selected variables for each reduction method and create a table.

    Parameters:
    - data_with_dummies (pd.DataFrame): The dataframe containing all columns.
    - initial_vars (list): The predictor variables in the initial model.
    - vif_vars (list): The variables after VIF filtering.
    - reduced_results_dict (dict): A dictionary containing reduced models and selected variables.
    - config (module): Configuration file containing paths and method information.

    Returns:
    - variable_selection_table (pd.DataFrame): A dataframe containing the selected variables across methods.
    """
    variables = set(data_with_dummies.columns)  # All possible variables
    variable_selection_dict = {}

    # Create rows for each variable and mark presence in Initial, VIF, Lasso, and Backward Elimination
    for var in variables:
        variable_selection_dict[var] = {
            'Initial': '✓' if var in initial_vars else '',
            'VIF': '✓' if var in vif_vars else '',
            'Lasso': '✓' if 'lasso' in reduced_results_dict and var in reduced_results_dict['lasso'][1] else '',
            'Backward Elimination': '✓' if 'backward_elimination' in reduced_results_dict and var in reduced_results_dict['backward_elimination'][1] else ''
        }

    # Convert the dictionary to a DataFrame
    variable_selection_table = pd.DataFrame.from_dict(variable_selection_dict, orient='index',
                                                      columns=['Initial', 'VIF', 'Lasso', 'Backward Elimination'])
    # Save the table using the utility function
    save_variable_selection(variable_selection_table, config)

    print("\n### Variable Selection and Model Information Table ###")
    print(variable_selection_table)

    return variable_selection_table

def save_metrics_and_comparisons(initial_model, reduced_results_dict, y):
    """
    Save model metrics and comparisons between the initial and reduced models.

    Parameters:
    - initial_model: The initial regression model.
    - reduced_results_dict: A dictionary containing reduced models.
    - y: The actual target variable data.

    Returns:
    - None
    """
    model_metrics_dict = {}

    for method, (reduced_model, _) in reduced_results_dict.items():
        if isinstance(reduced_model, sm.regression.linear_model.RegressionResultsWrapper):
            model_metrics_dict[method] = {
                'R-squared': reduced_model.rsquared,
                'MSE': reduced_model.mse_total,
                'P-values': reduced_model.pvalues.to_dict(),
                'Coefficients': reduced_model.params.to_dict()
            }

    # Save model metrics for all methods
    save_model_metrics(model_metrics_dict)

def calculate_and_save_rse(initial_model, reduced_results_dict, y, config):
    """
    Calculate and save the Residual Standard Error (RSE) for both initial and reduced models.

    Parameters:
    - initial_model: The initial regression model.
    - reduced_results_dict: A dictionary containing reduced models.
    - y: The actual target variable data.
    - config: Configuration file with paths to save results.

    Returns:
    - None
    """
    # Calculate and save RSE for the initial model
    residuals_initial = y - initial_model.fittedvalues
    rse_initial = (sum(residuals_initial ** 2) / (len(y) - len(initial_model.params))) ** 0.5
    print(f"\nInitial Model RSE: {rse_initial}")

    # Loop through reduced models and calculate their RSE
    for method, (reduced_model, _) in reduced_results_dict.items():
        if isinstance(reduced_model, sm.regression.linear_model.RegressionResultsWrapper):
            residuals_reduced = y - reduced_model.fittedvalues
            rse_reduced = (sum(residuals_reduced ** 2) / (len(y) - len(reduced_model.params))) ** 0.5
            print(f"Reduced Model ({method.capitalize()}) RSE: {rse_reduced}")

