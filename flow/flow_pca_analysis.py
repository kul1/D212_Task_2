import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from utils.visualizations import save_class_distribution, save_auc_curve, save_confusion_matrix, create_visualizations, save_scree_plot

RESULTS_DIR = "results/pca_analysis"
VISUALS_DIR = os.path.join(RESULTS_DIR, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

# Dummy configuration to replace hardcoded paths
class Config:
    TARGET_COLUMN = "Overweight_Yes"
    CONTINUOUS_COLUMNS = ['Age', 'Income', 'VitD_levels']  # Dynamically define these
    PCA_COMPONENTS_RETAINED = 3
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    KNN_CONFIG = {'n_neighbors': 5}

config = Config()


def visualize_standardization(data, numeric_columns, scaled_data, visuals_dir):
    """
    Create visualizations comparing original and standardized data.
    """
    os.makedirs(visuals_dir, exist_ok=True)

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


def save_scree_plot_with_elbow(explained_variance_ratio, visuals_dir):
    """
    Generate and save a scree plot with an elbow point.
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    elbow_point = np.argmax(cumulative_variance >= 0.8) + 1  # First PC to explain >= 80%

    scree_plot_path = os.path.join(visuals_dir, "scree_plot_with_elbow.png")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Individual Variance')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='x', linestyle='--', label='Cumulative Variance')
    plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow Point: {elbow_point}')
    plt.title('Scree Plot with Elbow Point')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.tight_layout()
    plt.savefig(scree_plot_path)
    plt.close()
    print(f"Scree plot with elbow point saved to: {scree_plot_path}")
    return elbow_point


def run_pca_analysis(data, config):
    """
    Perform PCA analysis on the dataset, generate relevant plots and metrics, and evaluate using a KNN classifier.
    """
    print("\n### Step 1: Standardizing Data ###")

    # Select continuous columns for PCA and scale them
    X = data[config.CONTINUOUS_COLUMNS]
    y = data[config.TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n### Visualizing Standardization ###")
    visualize_standardization(data, config.CONTINUOUS_COLUMNS, X_scaled, VISUALS_DIR)

    print("\n### Step 2: Performing PCA ###")
    pca = PCA(n_components=config.PCA_COMPONENTS_RETAINED)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Save the scree plot
    scree_plot_path = os.path.join(VISUALS_DIR, "scree_plot_with_elbow.png")
    save_scree_plot_with_elbow(explained_variance_ratio, VISUALS_DIR)

    print("\nExplained Variance by Component:")
    for i, var in enumerate(explained_variance_ratio, 1):
        print(f"PC{i}: {var:.4f} ({cumulative_variance[i-1]:.4f} cumulative)")

    explained_variance_path = os.path.join(VISUALS_DIR, "explained_variance.png")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', label='Cumulative Variance')
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, label='Individual Variance')
    plt.title('Explained Variance and Cumulative Variance by PCA Components')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.tight_layout()
    plt.savefig(explained_variance_path)
    plt.close()
    print(f"Explained variance plot saved to: {explained_variance_path}")

    print("\n### Step 3: Principal Component Loadings ###")
    for i, component in enumerate(pca.components_[:config.PCA_COMPONENTS_RETAINED], 1):
        print(f"PC{i} Loadings:\n{component}")

    print("\n### Step 4: Splitting Data for Classification ###")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y,
                                                        test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    knn = KNeighborsClassifier(n_neighbors=config.KNN_CONFIG['n_neighbors'])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    class_distribution_path = os.path.join(VISUALS_DIR, "class_distribution.png")
    save_class_distribution(data, config.TARGET_COLUMN, output_path=class_distribution_path)
    print(f"Class distribution saved to: {class_distribution_path}")

    try:
        if hasattr(knn, "predict_proba"):
            y_scores = knn.predict_proba(X_test)[:, 1]
            auc_curve_path = os.path.join(VISUALS_DIR, "auc_curve.png")
            save_auc_curve(y_test, y_scores, output_path=auc_curve_path)
            print(f"AUC curve saved to: {auc_curve_path}")
    except Exception as e:
        print(f"Error generating AUC curve: {e}")

    try:
        actual_labels = sorted(set(y_test))
        conf_matrix_path = os.path.join(VISUALS_DIR, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, output_path=conf_matrix_path, class_labels=actual_labels)
        print(f"Confusion matrix saved to: {conf_matrix_path}")

        class_report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        class_report = classification_report(y_test, y_pred, target_names=[str(label) for label in actual_labels])
        with open(class_report_path, "w") as f:
            f.write(class_report)
        print(f"Classification Report saved to: {class_report_path}")
    except Exception as e:
        print(f"Error generating evaluation metrics: {e}")

    print("\n### Step 6: Generating Visualizations ###")
    try:
        create_visualizations(data, target_column=config.TARGET_COLUMN, independent_variables=config.CONTINUOUS_COLUMNS)
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    print("\n### PCA Analysis Completed Successfully ###")
