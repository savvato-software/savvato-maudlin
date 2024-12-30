import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def generate(config, data_dir, model, X_train, y_train, X_test, y_true, y_preds):

#def generate_precision_recall_curve(config, data_dir, y_true, y_scores):
    """
    Plot the Precision-Recall Curve, calculate the best F1 threshold, and save the figure.

    Args:
        config (dict): Configuration dictionary.
        data_dir (str): Directory to save the output plot.
        y_true (array): True binary labels.
        y_scores (array): Predicted probabilities for the positive class.
    """
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
    avg_precision = average_precision_score(y_true, y_preds)

    # Compute F1 scores for each threshold, avoiding invalid values
    valid_mask = (precision + recall) > 0  # Mask invalid entries
    f1_scores = np.zeros_like(precision)  # Default F1 scores to 0
    f1_scores[valid_mask] = 2 * (precision[valid_mask] * recall[valid_mask]) / (precision[valid_mask] + recall[valid_mask])

    # Find the threshold with the maximum F1 score
    best_idx = np.argmax(f1_scores)

    # Adjust threshold lookup to avoid index mismatch
    if best_idx == 0:  # Handle edge case for first point
        best_threshold = thresholds[0]
    else:
        best_threshold = thresholds[best_idx - 1]  # Correct alignment

    best_f1 = f1_scores[best_idx]

    # Print the best threshold and F1-score to the console
    print(f"Best Threshold for F1: {best_threshold:.7f}")
    print(f"Best F1-Score: {best_f1:.7f}")

    # Plot the Precision-Recall curve
    output_path = data_dir + "/precision_recall_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.7f}')

    # Highlight the best F1 point
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f'Best F1 @ {best_threshold:.7f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid()

    # Save the plot
    plt.savefig(output_path)
    plt.close()

    print(f"Precision-Recall curve saved. firefox {output_path}")

    return best_threshold


