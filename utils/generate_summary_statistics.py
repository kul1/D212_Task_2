import pandas as pd

def generate_summary_statistics(data, target_column, continuous_columns, transformed_columns):
    """
    Generate summary statistics for the dependent and independent variables.

    Parameters:
    - data (pd.DataFrame): The cleaned dataset with dummy variables.
    - target_column (str): The name of the dependent variable (TotalCharge).
    - continuous_columns (list): List of continuous independent variable column names.
    - transformed_columns (list): List of dummy variable column names from the config.

    Returns:
    None
    """

    # Summary statistics for the target variable (dependent variable)
    print("\n### Summary Statistics for Dependent Variable (TotalCharge) ###")
    target_summary = data[target_column].describe()
    print(target_summary)

    # Summary statistics for continuous independent variables
    print("\n### Summary Statistics for Continuous Independent Variables ###")
    continuous_summary = data[continuous_columns].describe()
    print(continuous_summary)

    # Summary statistics for categorical (transformed) independent variables
    print("\n### Summary Statistics for Transformed Categorical Independent Variables ###")
    for column in transformed_columns:
        if column != target_column and column in data.columns:
            print(f"\nSummary for {column}:")
            print(data[column].value_counts())
        else:
            print(f"Warning: {column} not found in data.")

