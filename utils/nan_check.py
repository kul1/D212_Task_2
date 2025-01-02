# File: utils/nan_check.py

def final_nan_check(data, vif_filtered_vars, target_var, debug_mode=False):
    """
    Perform a final check for NaNs in the predictor and target variables before running reduction methods.

    Parameters:
    - data (pd.DataFrame): The dataset containing the predictor and target variables.
    - vif_filtered_vars (list): List of predictor variables filtered by VIF.
    - target_var (str): The target variable for the regression.
    - debug_mode (bool): If True, logs detailed debug information.

    Raises:
    - ValueError: If NaNs are found in predictor or target variables.
    """
    print("\n### Final Check for NaNs Before Running Reduction Methods ###")

    # Check for NaNs in predictor variables
    nan_predictors = data[vif_filtered_vars].isna().sum()
    nan_target = data[target_var].isna().sum()

    if debug_mode:
        print(f"NaNs in predictor variables:\n{nan_predictors}")
        print(f"NaNs in target variable:\n{nan_target}")

    # Raise an error if NaNs are found
    if nan_predictors.sum() > 0 or nan_target > 0:
        raise ValueError("NaN values found in predictor or target variables. Reduction methods cannot proceed.")

