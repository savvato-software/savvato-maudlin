
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def generate(config, data_dir, X_train, X_resampled, y_resampled, columns, pca):
    
    print("Generating boxplot output data...")

    # Create DataFrame
    df = pd.DataFrame(X_resampled, columns=columns)

    # Generate Boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.title("Box Plot")
    plt.savefig(os.path.join(data_dir, "boxplot.png"))
    plt.close()

    # Calculate Metrics
    metrics = {
        "mean": df.mean().to_dict(),
        "median": df.median().to_dict(),
        "std_dev": df.std().to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "correlation_matrix": df.corr().to_dict()
    }

    # Save metrics to JSON
    metrics_path = os.path.join(data_dir, "metrics.json")
    with open(metrics_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    print(f"Metrics saved to {metrics_path}")

