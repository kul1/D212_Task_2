import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA


def ensure_directory(output_path):
    """
    Ensure that the directory for the given path exists.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def save_confusion_matrix(y_true, y_pred, config, step="pca_analysis", class_labels=None):
    """
    Save a confusion matrix plot to the specified path derived from config.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - config: Configuration object.
    - step: Step name for visualization directory (optional).
    - class_labels: List of class labels for the confusion matrix (optional).
    """
    output_path = os.path.join(config.RESULTS_DIR, "visuals", step, "confusion_matrix.png")
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


def save_class_distribution(data, target_column, config, step="pca_analysis"):
    """
    Save a visualization of the class distribution for the target column.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - target_column: The target column to analyze.
    - config: Configuration object.
    - step: Step name for visualization directory (optional).
    """
    output_path = os.path.join(config.RESULTS_DIR, "visuals", step, "class_distribution.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    class_counts = data[target_column].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title(f"Class Distribution of {target_column}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Class distribution visualization saved to: {output_path}")

def create_visualizations(data, target_column, independent_variables, config, step="pca_analysis"):
    """
    Generate visualizations for exploratory data analysis.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - target_column: The target column.
    - independent_variables: List of independent variables.
    - config: Configuration object.
    - step: Specific analysis step (optional).
    """
    visuals_dir = os.path.join(config.RESULTS_DIR, "visuals", step)
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
    save_class_distribution(data, target_column, config, step)

    # Histograms for independent variables
    for var in independent_variables:
        if var in data.columns:
            hist_path = os.path.join(visuals_dir, f"{var}_distribution.png")
            ensure_directory(hist_path)
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


def save_auc_curve(y_true, y_scores, config, step="pca_analysis"):
    """
    Save an AUC (Area Under the Curve) plot to the specified path.

    Parameters:
    - y_true: Ground truth binary labels.
    - y_scores: Predicted probabilities or scores.
    - config: Configuration object.
    - step: Step name for visualization directory (optional).
    """
    output_path = os.path.join(config.RESULTS_DIR, "visuals", step, "auc_curve.png")
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


def visualize_standardization(data, numeric_columns, scaled_data, config, step="pca_analysis"):
    """
    Create visualizations comparing original and standardized data.

    Parameters:
    - data (pd.DataFrame): The original dataset.
    - numeric_columns (list): List of numeric columns to visualize.
    - scaled_data (np.array): Standardized data.
    - config: Configuration object.
    - step: Specific analysis step (optional).
    """
    visuals_dir = os.path.join(config.RESULTS_DIR, "visuals", step)
    ensure_directory(visuals_dir)

    # Original distribution
    original_distribution_path = os.path.join(visuals_dir, "original_distribution.png")
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(numeric_columns):
        plt.hist(data[column], bins=30, alpha=0.7, label="Original", color="blue")
        plt.title(f"Original {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(original_distribution_path)
    plt.close()
    print(f"Original distribution saved to: {original_distribution_path}")

    # Standardized distribution
    standardized_distribution_path = os.path.join(visuals_dir, "standardized_distribution.png")
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(numeric_columns):
        plt.hist(scaled_data[:, i], bins=30, alpha=0.7, label="Standardized", color="orange")
        plt.title(f"Standardized {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(standardized_distribution_path)
    plt.close()
    print(f"Standardized distribution saved to: {standardized_distribution_path}")


def plot_scree_plot(data, config, step="pca_analysis"):
    """
    Generate a scree plot showing explained variance by principal components.

    Parameters:
    - data (np.array): The dataset to perform PCA on.
    - config: Configuration object.
    - step: Step name for visualization directory (optional).
    """
    output_path = os.path.join(config.RESULTS_DIR, "visuals", step, "scree_plot.png")
    ensure_directory(output_path)

    pca = PCA()
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker="o", linestyle="--", color="blue")
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Scree plot saved to: {output_path}")
