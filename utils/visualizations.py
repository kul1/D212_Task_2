import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA

def ensure_directory(output_path):
    """Ensure that the directory for the given path exists."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

def visualize_standardization(data, numeric_columns, scaled_data, visuals_dir):
    """
    Create visualizations comparing original and standardized data.

    Parameters:
    - data (pd.DataFrame): The original dataset.
    - numeric_columns (list): List of numeric columns to visualize.
    - scaled_data (np.array): Standardized data.
    - visuals_dir (str): Path to save the visualizations.
    """
    ensure_directory(visuals_dir)

    # Original distribution
    original_distribution_path = os.path.join(visuals_dir, "original_distribution.png")
    plt.figure(figsize=(12, 6))
    for column in numeric_columns:
        data[column].plot(kind='density', alpha=0.7, label=column)
    plt.title("Original Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(original_distribution_path)
    plt.close()
    print(f"Original distribution saved to: {original_distribution_path}")

    # Standardized distribution
    standardized_distribution_path = os.path.join(visuals_dir, "standardized_distribution.png")
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(numeric_columns):
        plt.hist(scaled_data[:, i], bins=30, alpha=0.7, label=column)
    plt.title("Standardized Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(standardized_distribution_path)
    plt.close()
    print(f"Standardized distribution saved to: {standardized_distribution_path}")

def save_confusion_matrix(y_true, y_pred, output_path, class_labels=None):
    """Save a confusion matrix plot to the specified path."""
    ensure_directory(output_path)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def save_scree_plot(explained_variance_ratio, elbow_point, output_path):
    """
    Generate and save a scree plot with an elbow point.

    Parameters:
    - explained_variance_ratio (list): List of variance explained by each principal component.
    - elbow_point (int): The index of the elbow point (number of components to retain).
    - output_path (str): File path to save the scree plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Individual Variance')
    plt.axvline(x=elbow_point, color='red', linestyle='--', label='Elbow Point')
    plt.title('Scree Plot with Elbow Point')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Scree plot saved to: {output_path}")

def save_class_distribution(data, target_column, output_path):
    """
    Save a visualization of the class distribution for the target column.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - target_column: The target column to analyze.
    - output_path: Path to save the output visualization.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    class_counts = data[target_column].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        dodge=False,
        palette="viridis",
        legend=False,
    )
    plt.title(f"Class Distribution of {target_column}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Class distribution visualization saved to: {output_path}")

def create_visualizations(data, target_column, independent_variables):
    """Generate visualizations for exploratory data analysis."""
    # Hardcoded visuals directory
    visuals_dir = "results/pca_analysis/visuals"
    ensure_directory(visuals_dir)

    # Correlation heatmap
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        heatmap_path = os.path.join(visuals_dir, "correlation_heatmap.png")
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Correlation heatmap saved to: {heatmap_path}")
    else:
        print("No numeric columns available for correlation heatmap.")

    # Class distribution
    class_distribution_path = os.path.join(visuals_dir, "class_distribution.png")
    save_class_distribution(data, target_column, class_distribution_path)

    # Histograms for independent variables
    for var in independent_variables:
        if var in data.columns:
            hist_path = os.path.join(visuals_dir, f"{var}_distribution.png")
            ensure_directory(visuals_dir)  # Ensure the directory for each variable
            plt.figure(figsize=(8, 6))
            data[var].hist(bins=30, color="blue", alpha=0.7)
            plt.title(f"Distribution of {var}")
            plt.xlabel(var)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
            print(f"Distribution plot for '{var}' saved to: {hist_path}")
        else:
            print(f"Independent variable '{var}' not found in the dataset.")

    print("Visualizations generated successfully.")

def save_auc_curve(y_true, y_scores, output_path):
    """Save an AUC plot to the specified path."""
    ensure_directory(output_path)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"AUC curve saved to: {output_path}")
