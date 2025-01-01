
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
import json

def generate(config, data_dir, X_train, X_resampled, y_resampled, columns, pca):

    print("Generating oversampling output data...")

    oversampling_method = config['pre_training']['oversampling'].get('method', '').lower()

    X_combined = np.vstack((X_train, X_resampled))
    y_combined = np.hstack((np.zeros(len(X_train)), np.ones(len(X_resampled))))

    title = oversampling_method.upper() + " Synthetic Data"

    if config['pre_training']['oversampling']['calculate_long_running_diagrams']:
        plot_data(config, data_dir, X_combined, y_combined, title, 'tsne')
    else:
        plot_data(config, data_dir, X_combined, y_combined, title, 'pca', pca)

def plot_data(config, data_dir, X, y, title, method='pca', pca=None):
    reducer = pca if method == 'pca' else TSNE(n_components=2, random_state=42, perplexity=30)
    X_reduced = reducer.fit_transform(X) if method == 'tsne' or pca is None else pca.transform(X)

    centroid_distance = None
    silhouette = None
    min_distance = None
    max_distance = None
    avg_distance = None

    # Compute metrics
    if config['pre_training']['oversampling']['calculate_long_running_diagrams']:

        centroid_distance = np.linalg.norm(np.mean(X_reduced[y == 0], axis=0) - np.mean(X_reduced[y == 1], axis=0))
        silhouette = silhouette_score(X_reduced, y)
        distances = pairwise_distances(X_reduced)
    
        min_distance = np.min(distances[y == 0][:, y == 1])
        max_distance = np.max(distances[y == 0][:, y == 1])
        avg_distance = np.mean(distances[y == 0][:, y == 1])

    # Variance explained (PCA only)
    variance_explained = None
    cumulative_variance = None
    if method == 'pca' and isinstance(reducer, PCA):
        variance_explained = reducer.explained_variance_ratio_[:2].tolist()
        cumulative_variance = sum(reducer.explained_variance_ratio_[:2])

    # Density metrics
    density_original = len(X[y == 0]) / np.ptp(X_reduced[y == 0], axis=0).prod()
    density_synthetic = len(X[y == 1]) / np.ptp(X_reduced[y == 1], axis=0).prod()

    # Outlier metrics
    outlier_threshold = 1.5 * np.std(X_reduced)
    outliers_original = np.sum(np.linalg.norm(X_reduced[y == 0] - np.mean(X_reduced[y == 0], axis=0), axis=1) > outlier_threshold)
    outliers_synthetic = np.sum(np.linalg.norm(X_reduced[y == 1] - np.mean(X_reduced[y == 1], axis=0), axis=1) > outlier_threshold)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], label='Original Data', alpha=0.5, s=10, color='blue')
    plt.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], label='Synthetic Data', alpha=0.5, s=10, color='red')
    plt.title(f'{title} - {method.upper()} Projection')
    plt.legend()
    plot_path = os.path.join(data_dir, f"smote_vs_original_{method}.png")
    plt.savefig(plot_path)
    plt.close()

    # Prepare textual description for the JSON file
    description = {
        "title": title,
        "method": method.upper(),
        "original_data_color": "blue",
        "synthetic_data_color": "red",
        "output_image": os.path.basename(plot_path),
        "metrics": {
            "cluster_overlap": {
                "centroid_distance": float(centroid_distance) if centroid_distance is not None else None,
                "silhouette_score": float(silhouette) if silhouette is not None else None
            },
            "separation": {
                "min_distance": float(min_distance) if min_distance is not None else None,
                "max_distance": float(max_distance) if max_distance is not None else None,
                "avg_distance": float(avg_distance) if avg_distance is not None else None
                },
            "variance_explained": {
                "component_1": variance_explained[0] if variance_explained else None,
                "component_2": variance_explained[1] if variance_explained else None,
                "cumulative": float(cumulative_variance) if cumulative_variance else None
            },
            "density": {
                "original_density": float(density_original),
                "synthetic_density": float(density_synthetic),
                },
            "outliers": {
                "original": int(outliers_original),
                "synthetic": int(outliers_synthetic)
                }
        }
    }

    # Save textual description as JSON
    json_path = os.path.join(data_dir, f"smote_vs_original_{method}_description.json")
    with open(json_path, 'w') as json_file:
        json.dump(description, json_file, indent=4)

    print(f"Visualization saved at: {plot_path}")
    print(f"Description saved at: {json_path}")
   
