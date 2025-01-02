# utils/variable_selection.py

import pandas as pd

def collect_variables(config, vif_module, lasso_module, backward_elimination_module):
    """
    Collect variables from other submodules.

    Args:
        config: The configuration file containing initial variables.
        vif_module: Module providing VIF-filtered variables.
        lasso_module: Module providing Lasso-selected variables.
        backward_elimination_module: Module providing backward elimination-selected variables.

    Returns:
        A dictionary with initial, VIF, Lasso, and Backward Elimination variables.
    """
    # Collect initial variables from the configuration
    initial_vars = list(config.COLUMN_CONFIG.keys())

    # Get variables from other modules
    vif_filtered_vars = vif_module.get_vif_filtered_variables()
    lasso_selected_vars = lasso_module.get_lasso_selected_variables()
    backward_elimination_vars = backward_elimination_module.get_backward_elimination_variables()

    # Gather variables into a dictionary
    selected_vars_dict = {
        'initial': initial_vars,
        'vif': vif_filtered_vars,
        'lasso': lasso_selected_vars,
        'backward_elimination': backward_elimination_vars
    }

    return selected_vars_dict
import pandas as pd

def display_variable_selection_table(initial_vars, vif_vars, reduced_vars_dict):
    """
    Display a table of selected variables from different reduction methods.

    Parameters:
    - initial_vars (list): The initial list of variables.
    - vif_vars (list): The variables filtered based on VIF.
    - reduced_vars_dict (dict): A dictionary of variables selected by each reduction method.

    """
    # Find the maximum length among all variable lists
    max_length = max(len(initial_vars), len(vif_vars), *[len(vars) for vars in reduced_vars_dict.values()])

    # Pad all lists to the maximum length with empty strings or None
    def pad_list(var_list, length):
        return var_list + [''] * (length - len(var_list))

    initial_vars_padded = pad_list(initial_vars, max_length)
    vif_vars_padded = pad_list(vif_vars, max_length)

    # Create a dictionary for all the variable selections
    selection_data = {
        'Initial': initial_vars_padded,
        'VIF': vif_vars_padded
    }

    # Add entries for each method in the reduced_vars_dict
    for method_name, vars_list in reduced_vars_dict.items():
        selection_data[method_name] = pad_list(vars_list, max_length)

    # Create the DataFrame and display it
    selection_df = pd.DataFrame(selection_data)
    print("\n### Step 13: Variable Selection Table ###")
    print(selection_df)

