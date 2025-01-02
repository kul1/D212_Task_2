# File Path: utils/variable_definitions.py

def define_target_and_predictors(data_with_dummies, config):
    """Defines the target variable and predictor variables based on config."""
    target_var = config.TARGET_COLUMN
    predictor_vars = [col for col in data_with_dummies.columns if col != target_var]
    return target_var, predictor_vars

