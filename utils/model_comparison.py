# File: utils/model_comparison.py

import statsmodels.api as sm
import numpy as np

def calculate_mse(model, y_true):
    """
    Calculate Mean Squared Error (MSE) for a given model manually.

    Parameters:
    - model: The fitted regression model (from OLS).
    - y_true: The actual target values.

    Returns:
    - mse: Mean Squared Error value.
    """
    if hasattr(model, 'fittedvalues'):
        residuals = y_true - model.fittedvalues
        rss = np.sum(np.square(residuals))  # Residual Sum of Squares
        mse = rss / len(y_true)  # Mean Squared Error
        return mse
    else:
        print("Warning: Model does not have fitted values. Unable to calculate MSE.")
        return None
# File: utils/model_comparison.py

def compare_models(initial_model, reduced_results_dict, y_true):
    """
    Compare the initial model and all reduced models based on R-squared, MSE, and optionally other metrics.

    Parameters:
    - initial_model: The full regression model before reduction.
    - reduced_results_dict: A dictionary of reduced models and their selected variables.
    - y_true: The actual target values.
    """
    initial_r2 = initial_model.rsquared
    initial_mse = calculate_mse(initial_model, y_true)

    for method_name, (reduced_model, selected_variables) in reduced_results_dict.items():
        print(f"\n### Comparing Initial Model with Reduced Model: {method_name.capitalize()} ###")

        if isinstance(reduced_model, sm.regression.linear_model.RegressionResultsWrapper):
            reduced_r2 = reduced_model.rsquared
            reduced_mse = calculate_mse(reduced_model, y_true)

            print(f"\n### Comparison of Models ###")
            print(f"Initial Model R-squared: {initial_r2:.4f}, MSE: {initial_mse:.4f}")
            print(f"Reduced Model (by {method_name}) R-squared: {reduced_r2:.4f}, MSE: {reduced_mse:.4f}")

            # Additional details omitted for brevity
        else:
            print(f"\n### Reduced Model (by {method_name}) is not a valid regression model.")

