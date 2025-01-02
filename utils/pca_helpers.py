import pandas as pd

def save_pca_loadings(pca, feature_names, output_path):
    """Save PCA loadings to a file."""
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    loadings.to_csv(output_path, index=True)
    print(f"PCA Loadings saved to: {output_path}")
