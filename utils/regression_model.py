# File Path: utils/regression_model.py

import statsmodels.api as sm

def run_initial_regression(data_with_dummies, predictor_vars, target_var):
    """Runs the initial regression model."""
    X_initial = sm.add_constant(data_with_dummies[predictor_vars])
    y = data_with_dummies[target_var]
    initial_model = sm.OLS(y, X_initial).fit()
    print("\n### Initial Regression Results ###\n")
    print(initial_model.summary())
    return initial_model, y

