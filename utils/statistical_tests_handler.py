# statistical_tests_handler.py

import os
import pandas as pd
from utils.statistical_tests import run_chi_square_test, run_pearson_correlation, run_t_test

def handle_statistical_tests(data_with_dummies, target_var, categorical_vars, numeric_vars, config):
    """
    Run Chi-Square, Pearson Correlation, and T-tests, save results, and print save locations.

    Parameters:
    - data_with_dummies: The DataFrame with the predictors and target variable.
    - target_var: The target variable.
    - categorical_vars: List of categorical variables for the Chi-Square test.
    - numeric_vars: List of numeric variables for Pearson correlation.
    - config: Configuration object for output paths.

    Returns:
    - None: Saves the results and prints their save locations.
    """
    results_dir = config.RESULTS_DIR

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run Chi-Square test for categorical variables
    chi_square_results = run_chi_square_test(data_with_dummies, target_var, categorical_vars)
    chi_square_file = os.path.join(results_dir, 'chi_square_results.csv')
    pd.DataFrame(chi_square_results).transpose().to_csv(chi_square_file)
    print(f"\n### Chi-Square Test Results ###\n")
    print(pd.DataFrame(chi_square_results).transpose())
    print(f"\nChi-Square Test results saved to: {chi_square_file}")

    # Run Pearson correlation for numeric variables
    pearson_results = run_pearson_correlation(data_with_dummies, numeric_vars, target_var)
    pearson_file = os.path.join(results_dir, 'pearson_correlation_results.csv')
    pd.DataFrame(pearson_results).transpose().to_csv(pearson_file)
    print(f"\n### Pearson Correlation Results ###\n")
    print(pd.DataFrame(pearson_results).transpose())
    print(f"\nPearson Correlation results saved to: {pearson_file}")

    # Optionally run a T-test for numeric variables against a specific group
    t_stat, p_value = run_t_test(data_with_dummies, 'Income', 'Overweight_Yes')
    t_test_file = os.path.join(results_dir, 't_test_results.txt')
    with open(t_test_file, 'w') as f:
        f.write(f"T-test for Income by Overweight group: t-stat = {t_stat}, p-value = {p_value}\n")
    print(f"\n### T-test Results ###\n")
    print(f"T-test for Income by Overweight group: t-stat = {t_stat}, p-value = {p_value}")
    print(f"\nT-test results saved to: {t_test_file}")


