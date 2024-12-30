
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate(config, data_dir, X_train, X_resampled, y_resampled, columns, pca):
    print("Generating correlation matrix output data...")

    # Create DataFrame
    df = pd.DataFrame(X_resampled, columns=columns)

    # Compute Correlation Matrix
    corr_matrix = df.corr()

    # Flatten the correlation matrix to get pairs
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    corr_pairs = corr_pairs[corr_pairs['Feature1'] != corr_pairs['Feature2']]  # Remove self-correlations

    # Remove duplicate pairs (e.g., A-B and B-A)
    corr_pairs['OrderedPair'] = corr_pairs.apply(lambda x: tuple(sorted([x['Feature1'], x['Feature2']])), axis=1)
    corr_pairs = corr_pairs.drop_duplicates(subset=['OrderedPair']).drop(columns=['OrderedPair'])

    # Find max, min, and average correlations
    max_corr_row = corr_pairs.loc[corr_pairs['Correlation'].idxmax()]
    min_corr_row = corr_pairs.loc[corr_pairs['Correlation'].idxmin()]
    avg_correlation = float(corr_pairs['Correlation'].mean())

    # Metrics Calculation
    metrics = {}
    metrics['max_correlation'] = [max_corr_row['Feature1'], max_corr_row['Feature2'], float(max_corr_row['Correlation'])]
    metrics['min_correlation'] = [min_corr_row['Feature1'], min_corr_row['Feature2'], float(min_corr_row['Correlation'])]
    metrics['avg_correlation'] = avg_correlation

    # Filter positive and negative correlations separately
    positive_corr_pairs = corr_pairs[corr_pairs['Correlation'] > 0]
    negative_corr_pairs = corr_pairs[corr_pairs['Correlation'] < 0]

    # Sort correlations
    top_5_highest = positive_corr_pairs.sort_values(by='Correlation', ascending=False).head(5)
    bottom_5_lowest = negative_corr_pairs.sort_values(by='Correlation').head(5)

    # Update metrics
    metrics['top_5_highest_correlations'] = top_5_highest.apply(lambda x: [x['Feature1'], x['Feature2'], x['Correlation']], axis=1).tolist()
    metrics['bottom_5_lowest_correlations'] = bottom_5_lowest.apply(lambda x: [x['Feature1'], x['Feature2'], x['Correlation']], axis=1).tolist()

    # Dynamically adjust figure size
    fig_width = max(10, 0.5 * len(columns))
    fig_height = max(8, 0.5 * len(columns))

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=columns, yticklabels=columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()  # Adjust layout to fit labels properly

    # Save the figure
    plt.savefig(os.path.join(data_dir, "correlation_matrix.png"))
    plt.close()

    # Save Metrics as JSON
    with open(os.path.join(data_dir, "correlation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

