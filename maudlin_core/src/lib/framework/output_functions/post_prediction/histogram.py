import os
import matplotlib.pyplot as plt

def generate(predictions, data_dir, ground_truth):
    plt.figure()
    plt.hist(predictions, bins=50, alpha=0.75, label='Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predictions')
    histogram_path = os.path.join(data_dir, 'histogram.png')
    plt.savefig(histogram_path)
    plt.close()
    return histogram_path


