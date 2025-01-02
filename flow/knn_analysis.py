import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans  # Add KMeans for Elbow Method
from utils.visualizations import create_visualizations, save_class_distribution, save_auc_curve, save_recommendations_summary

def run_knn_classification(data_with_dummies, config):
    # Fetch predictor variables and target variable directly from config
    predictor_vars = config.INDEPENDENT_VARIABLES  # Extract from config
    target_var = config.TARGET_COLUMN  # Extract from config

    # Step: Split the data into training and test sets
    X = data_with_dummies[predictor_vars]
    y = data_with_dummies[target_var]
    test_size = config.TEST_SIZE
    random_state = config.RANDOM_STATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 1: **Elbow Method** to determine the optimal number of clusters
    # Perform K-means clustering for a range of k values and calculate WCSS (Within-Cluster Sum of Squares)
    wcss = []
    for i in range(1, 11):  # Testing k from 1 to 10
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
        kmeans.fit(X)  # Using the predictor variables for clustering
        wcss.append(kmeans.inertia_)

    # Plot the WCSS values to find the "elbow"
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')

    # Save the plot instead of displaying it
    elbow_plot_path = os.path.join(config.RESULTS_DIR, 'elbow_method_plot.png')
    plt.savefig(elbow_plot_path)  # Save the plot to the results directory
    plt.close()  # Close the plot to avoid it blocking further execution

    print(f"Elbow method plot saved to: {elbow_plot_path}")

    # **Report the optimal number of clusters based on the elbow plot**
    optimal_clusters = 3  # This is based on your observation of the elbow plot. Update based on your findings.
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Step 2: **KNN Classifier** to predict on the test set
    knn = KNeighborsClassifier(
        n_neighbors=config.KNN_CONFIG['n_neighbors'],
        weights=config.KNN_CONFIG['weights'],
        algorithm=config.KNN_CONFIG['algorithm'],
        metric=config.KNN_CONFIG['metric']
    )
    knn.fit(X_train, y_train)

    # Step: Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    print("\n### Classification Model Evaluation ###")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and log accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    # Cross-validation (if specified)
    if config.KNN_CONFIG.get('cross_val', False):
        cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

    # AUC Curve calculation
    try:
        y_scores = knn.predict_proba(X_test)[:, 1]  # Get probability estimates
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.2f}")

        # Save AUC Curve
        auc_curve_path = os.path.join(config.RESULTS_DIR, 'auc_curve.png')
        save_auc_curve(y_test, y_scores, auc_curve_path)

    except Exception as e:
        print("Model is not defined. Cannot generate AUC curve.")
        print(f"Error: {e}")

    # Generate Visualizations
    create_visualizations(data_with_dummies, target_var, predictor_vars, conf_matrix, report, config, step='knn_classification', model=knn)

    # Save Class Distribution
    class_distribution_path = os.path.join(config.RESULTS_DIR, 'class_distribution.png')
    save_class_distribution(data_with_dummies, target_var, class_distribution_path)

    # Save Recommendations Summary
    recommendations_path = os.path.join(config.RESULTS_DIR, 'recommendations_summary.png')
    save_recommendations_summary(recommendations_path)
