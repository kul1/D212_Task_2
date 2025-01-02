# vif_calculation.py

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data, config):
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.
    """
    target_var = config.TARGET_COLUMN
    vif_threshold = config.VIF_THRESHOLD

    print(f"\n### Starting VIF Calculation ###")

    # Debug: Print current DataFrame columns
    print("Current DataFrame columns:", data.columns.tolist())

    # Check if target_var is in the DataFrame before attempting to drop
    if target_var in data.columns:
        predictors = data.drop(columns=[target_var])
        print(f"Dropping {target_var} from predictors.")
    elif f"{target_var}_Yes" in data.columns:
        predictors = data.drop(columns=[f"{target_var}_Yes"])
        print(f"Note: {target_var} not found in DataFrame columns. Using {target_var}_Yes instead.")
    else:
        print(f"Note: Neither {target_var} nor {target_var}_Yes found in DataFrame columns. Proceeding without dropping.")
        predictors = data

    vif_data = pd.DataFrame()
    vif_data['Variable'] = predictors.columns
    vif_data['VIF'] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

    print("\n### Calculated VIF Values for Each Predictor ###")
    print(vif_data)

    vif_filtered_vars = vif_data[vif_data['VIF'] <= vif_threshold]['Variable'].tolist()

    print(f"\n### Filtered Predictors with VIF <= {vif_threshold} ###\n{vif_filtered_vars}")

    return vif_filtered_vars, vif_data
