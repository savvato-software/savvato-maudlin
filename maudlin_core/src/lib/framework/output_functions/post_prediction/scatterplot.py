import os
import matplotlib.pyplot as plt

def generate(predictions, data_dir, ground_truth):
    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Scatterplot of Predictions vs. Ground Truth')
    scatter_path = os.path.join(data_dir, 'scatterplot.png')
    plt.savefig(scatter_path)
    plt.close()
    return scatter_path


