# statistical_tests.py

import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, ttest_ind

def run_chi_square_test(data, target_var, categorical_vars):
    """
    Run the Chi-Square test for independence between the target variable and categorical predictors.

    Parameters:
    - data: The DataFrame containing dummy variables and predictors.
    - target_var: The target variable for the regression model.
    - categorical_vars: List of categorical variables to test against the target variable.

    Returns:
    - chi_square_results: A dictionary of Chi-Square statistics and p-values for each categorical variable.
    """
    chi_square_results = {}

    for var in categorical_vars:
        contingency_table = pd.crosstab(data[var], data[target_var])
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        chi_square_results[var] = {
            'chi2_stat': chi2,
            'p_value': p,
            'degrees_of_freedom': dof
        }

    return chi_square_results


def run_pearson_correlation(data, numeric_vars, target_var):
    """
    Run Pearson correlation between the numeric variables and the target variable.

    Parameters:
    - data: The DataFrame containing numeric variables.
    - numeric_vars: List of numeric variables to test correlation.
    - target_var: Target variable for the correlation analysis.

    Returns:
    - pearson_results: A dictionary of Pearson correlation coefficients and p-values for each numeric variable.
    """
    pearson_results = {}

    for var in numeric_vars:
        corr, p_value = pearsonr(data[var], data[target_var])
        pearson_results[var] = {
            'correlation': corr,
            'p_value': p_value
        }

    return pearson_results


def run_t_test(data, numeric_var, group_var):
    """
    Run an independent T-test on a numeric variable between two groups.

    Parameters:
    - data: The DataFrame containing the variables.
    - numeric_var: The numeric variable to test.
    - group_var: The categorical variable to define groups for comparison.

    Returns:
    - t_stat: T-statistic
    - p_value: p-value from the T-test
    """
    group1 = data[data[group_var] == 0][numeric_var]
    group2 = data[data[group_var] == 1][numeric_var]

    t_stat, p_value = ttest_ind(group1, group2)

    return t_stat, p_value

