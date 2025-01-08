# utils/clean_and_create_data.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_encoding import create_dummies
from utils.data_loader import save_prepared_data_with_dummies
from data_cleaning import clean_data

def clean_and_create_data(data, config):
    """
    Clean the raw data including the target column and create dummy variables.

    Parameters:
    - data (pd.DataFrame): The raw dataset.
    - config: The configuration object containing column information.

    Returns:
    - pd.DataFrame: The cleaned and scaled dataset with dummy variables created.
    """
    print("### Step 1: Handle Missing Values ###")
    # Handle missing values for numerical columns
    numerical_columns = [col for col, col_type in config.COLUMN_CONFIG.items() if col_type == 'continuous']
    print(f"Numerical columns to clean: {numerical_columns}")
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Handle missing values for categorical columns
    categorical_columns = [col for col, col_type in config.COLUMN_CONFIG.items() if col_type == 'categorical']
    print(f"Categorical columns to clean: {categorical_columns}")
    for col in categorical_columns:
        data[col] = data[col].fillna('Unknown')  # Replace NaN with 'Unknown'

    print("Missing value handling completed.")

    print("### Step 2: Create Dummy Variables ###")
    # Create dummy variables for categorical columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    print("Dummy variables created.")

    print("### Step 3: Convert Boolean Columns to Integers ###")
    # Convert boolean columns to integers (0 and 1)
    boolean_columns = [col for col in data.columns if data[col].dtype == bool]
    print(f"Boolean columns to convert: {boolean_columns}")
    for col in boolean_columns:
        data[col] = data[col].astype(int)
    print("Boolean conversion completed.")

    print("### Step 4: Scale Continuous Variables ###")
    # Standardize continuous variables
    scaler = StandardScaler()
    scaled_columns = [col for col in data.columns if col in numerical_columns]
    print(f"Continuous columns to scale: {scaled_columns}")
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    print("Continuous variables scaled using StandardScaler.")

    print("### Step 5: Validate Data Cleaning ###")
    # Ensure there are no missing values
    assert not data.isnull().any().any(), "Data cleaning failed. NaN values remain."
    print("Data cleaning validated. No NaN values remain.")

    print("### Data Cleaning and Preparation Completed ###")

    print("### Displaying Sample of Cleaned Data ###")
    # Display a sample of cleaned data for verification
    print(data[scaled_columns].head())  # Show scaled continuous columns
    print(data.head())  # Show complete sample data

    return data
