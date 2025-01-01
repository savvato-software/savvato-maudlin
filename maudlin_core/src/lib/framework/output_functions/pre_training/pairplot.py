
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import skew, kurtosis

def generate(config, data_dir, X_train, X_resampled, y_resampled, columns, pca):

    print("Generating pairplot output data.....")

    # Create a DataFrame from resampled data
    df = pd.DataFrame(X_resampled, columns=columns)

    # Validate consistency between features and labels
    if len(df) != len(y_resampled):
        raise ValueError("Mismatch in number of samples between features and labels for pairplot.")

    # Add label column for visualization
    df['label'] = y_resampled

    # Generate pairplot
    pairplot = sns.pairplot(df, hue='label', diag_kind='kde')
    plot_path = os.path.join(data_dir, "pairplot.png")
    pairplot.savefig(plot_path)
    plt.close()

    # Compute metrics
    metrics = {}
    for column in columns:
        feature_data = df[column]
        metrics[column] = {
            "mean": feature_data.mean(),
            "variance": feature_data.var(),
            "skewness": skew(feature_data),
            "kurtosis": kurtosis(feature_data),
            "outliers": {
                "count": np.sum(np.abs((feature_data - feature_data.mean()) / feature_data.std()) > 3),
                "percentage": np.mean(np.abs((feature_data - feature_data.mean()) / feature_data.std()) > 3) * 100
            }
        }

    # Prepare textual description for the JSON file
    description = {
        "title": "Pairplot of Features",
        "x-axis": columns,
        "y-axis": columns,
        "hue": "label",
        "diag_kind": "kde",
        "output_image": "pairplot.png",
        "metrics": metrics
    }

    # Save textual description as JSON
    json_path = os.path.join(data_dir, "pairplot_description.json")
    with open(json_path, 'w') as json_file:
        json.dump(description, json_file, indent=4)

    print(f"Pairplot saved at: {plot_path}")
    print(f"Description saved at: {json_path}")

