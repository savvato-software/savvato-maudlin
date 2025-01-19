import pandas as pd
from sklearn.decomposition import PCA


def apply_pca_if_enabled(config, X_train, X_test, X_val, columns):
    """Applies PCA to the dataset if enabled in the configuration."""
    if not config['pca']['enabled']:
        print("NOT applying PCA to X... it is not enabled.")
        return None, None, None, None

    print("Applying PCA to X...")
    pca_config = config['pca']
    pca = PCA(n_components=pca_config['params'].get('n_components', 5))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = None
    X_val_pca = None

    # Check if X_test is not None before applying PCA
    if X_test is not None and len(X_test) > 0:
        X_test_pca = pca.transform(X_test)

    # Check if X_val is not None before applying PCA
    if X_val is not None and len(X_val) > 0:
        X_val_pca = pca.transform(X_val)

    if pca_config['console_logging']:
        log_pca_statistics(pca, columns)

    return X_train_pca, X_test_pca, X_val_pca, pca


def log_pca_statistics(pca, columns):
    """Logs PCA statistics such as explained variance and top features."""
    pca_components = pd.DataFrame(pca.components_, columns=columns)

    for i in range(pca.n_components):
        print(f"Principal Component {i+1}:")
        print(pca_components.iloc[i].abs().sort_values(ascending=False).head(5))
        print("-" * 40)

    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio:")
    for i, ratio in enumerate(explained_variance):
        print(f"Principal Component {i+1}: {ratio:.4f}")

    cumulative_variance = explained_variance.cumsum()
    print("\nCumulative Explained Variance:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"Up to Principal Component {i+1}: {cum_var:.4f}")


