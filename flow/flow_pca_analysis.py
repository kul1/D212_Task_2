import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from utils.visualizations import save_class_distribution, save_auc_curve, save_confusion_matrix, create_visualizations
from utils.config_loader import load_config

config = load_config()
print(f"Loaded configuration: {config}")
print(f"Config type: {type(config)}")

def run_pca_analysis(data, config):
    """
    Perform PCA analysis on the dataset, generate relevant plots and metrics, and evaluate using a KNN classifier.

    Parameters:
    - data: DataFrame with cleaned and preprocessed data.
    - config: Configuration object containing analysis parameters.
    """
    # Ensure config is correct
    if not hasattr(config, "RESULTS_DIR"):
        raise ValueError("Invalid config object. Ensure 'load_config' is called and config is passed properly.")
    print("\n### Step 1: Standardizing Data ###")

    # Select continuous columns for PCA and scale them
    numeric_columns = data.select_dtypes(include=['number']).columns
    X = data[numeric_columns]
    y = data[config.TARGET_COLUMN]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n### Step 2: Performing PCA ###")
    # Perform PCA
    pca = PCA(n_components=config.PCA_COMPONENTS_RETAINED)
    X_pca = pca.fit_transform(X_scaled)

    print("Data After PCA:")
    print(X_pca[:5])

    # Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("\nExplained Variance by Component:")
    for i, var in enumerate(explained_variance_ratio, 1):
        print(f"PC{i}: {var:.4f} ({cumulative_variance[i-1]:.4f} cumulative)")

    # Plot explained variance
    explained_variance_path = os.path.join(config.VISUALS_DIR, "explained_variance.png")
    os.makedirs(config.VISUALS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', label='Cumulative Variance')
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, label='Individual Variance')
    plt.title('Explained Variance and Cumulative Variance by PCA Components')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.savefig(explained_variance_path)
    plt.close()
    print(f"Explained variance plot saved to: {explained_variance_path}")

    print("\n### Step 3: Principal Component Loadings ###")
    for i, component in enumerate(pca.components_[:config.PCA_COMPONENTS_RETAINED], 1):
        print(f"PC{i} Loadings:\n{component}")

    # Train-Test Split and KNN Classification
    print("\n### Step 4: Splitting Data for Classification ###")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    knn = KNeighborsClassifier(n_neighbors=config.KNN_CONFIG['n_neighbors'])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save class distribution
    print(f"Before operation: {type(config)}, {config.RESULTS_DIR}")
    class_distribution_path = os.path.join(config.VISUALS_DIR, "class_distribution.png")
    os.makedirs(config.VISUALS_DIR, exist_ok=True)
    save_class_distribution(data, config.TARGET_COLUMN, class_distribution_path)
    print(f"Class distribution saved to: {class_distribution_path}")

    # Generate AUC curve
    try:
        if hasattr(knn, "predict_proba"):
            y_scores = knn.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            print(f"AUC: {roc_auc:.2f}")

            auc_curve_path = os.path.join(config.VISUALS_DIR, "auc_curve.png")
            save_auc_curve(y_test, y_scores, auc_curve_path)
    except Exception as e:
        print(f"Error generating AUC curve: {e}")

    # Generate confusion matrix and classification report
    print("\n### Step 5: Generating Evaluation Metrics ###")
    try:
        actual_labels = sorted(set(y_test))
        conf_matrix_path = os.path.join(config.VISUALS_DIR, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, conf_matrix_path, actual_labels)

        class_report = classification_report(y_test, y_pred, target_names=[str(label) for label in actual_labels])
        class_report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        class_report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
        with open(class_report_path, "w") as f:
            f.write(class_report)
        print(f"Classification Report saved to: {class_report_path}")
    except Exception as e:
        print(f"Error generating evaluation metrics: {e}")

    # Generate visualizations
    print("\n### Step 6: Generating Visualizations ###")
    try:
        create_visualizations(data, config.TARGET_COLUMN, config.CONTINUOUS_COLUMNS, config.VISUALS_DIR, step="pca_analysis")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    print("\n### PCA Analysis Completed Successfully ###")
