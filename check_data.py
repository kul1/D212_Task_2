import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/kul/Documents/wgu/D212/Task2_2/Task_2_program/prepared/pca_analysis/prepared_data_for_pca_analysis.csv"
data = pd.read_csv(file_path)

print(f"Shape of the dataset: {data.shape}")
print(f"Columns in the dataset: {list(data.columns)}")

# Display the first 5 rows as a sample
print("Sample data:")
print(data.head())

# 1. Check for Missing Values
print("\n### Checking for Missing Values ###")
missing_summary = data.isnull().sum()
missing_values = missing_summary[missing_summary > 0]
if missing_values.empty:
    print("No missing values found.")
else:
    print("Missing Values Summary:")
    print(missing_values)

# 2. Verify Consistency in Column Data Types
print("\n### Verifying Column Data Types ###")
print(data.dtypes)

# 3. Check for Outliers in Continuous Variables
print("\n### Checking for Outliers ###")
continuous_columns = ['Age', 'Income', 'VitD_levels']  # Add other continuous columns as needed
for column in continuous_columns:
    print(f"Visualizing outliers for {column}...")
    plt.hist(data[column], bins=30, alpha=0.7)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# 4. Verify Dummy Variable Creation
print("\n### Verifying Dummy Variables ###")
dummy_columns = [col for col in data.columns if col.endswith('_Yes') or col.endswith('_No')]
print(f"Dummy Variables: {dummy_columns}")

# 5. Check Dataset Dimensions
print("\n### Checking Dataset Dimensions ###")
expected_rows = 10000  # Adjust based on your original dataset size
actual_rows, actual_columns = data.shape
print(f"Expected rows: {expected_rows}, Actual rows: {actual_rows}")
print(f"Actual columns: {actual_columns}")

# 6. Check for Duplicates
print("\n### Checking for Duplicates ###")
duplicate_rows = data[data.duplicated()]
if duplicate_rows.empty:
    print("No duplicate rows found.")
else:
    print(f"Number of duplicate rows: {len(duplicate_rows)}")
    print("Sample duplicate rows:")
    print(duplicate_rows.head())

# 7. Validate Target Variable
print("\n### Validating Target Variable ###")
target_column = "Overweight_Yes"
if target_column in data.columns:
    unique_values = data[target_column].unique()
    print(f"Unique values in target column '{target_column}': {unique_values}")
    if set(unique_values) <= {0, 1}:
        print("Target column is binary and valid.")
    else:
        print("Error: Target column contains unexpected values.")
else:
    print(f"Error: Target column '{target_column}' not found in the dataset.")
