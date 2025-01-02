# File: utils/model_reduction_helpers.py

import pandas as pd
from utils.save_results import save_variable_selection

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
    # Create a set of all possible variables in the dataset
    variables = set(data_with_dummies.columns)
    variable_selection_dict = {}

    # Create rows for each variable and mark its presence in Initial, VIF, Lasso, and Backward Elimination
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

