import os
import matplotlib.pyplot as plt

def generate(predictions, data_dir, ground_truth):
    plt.figure()
    plt.plot(predictions, label='Predictions', alpha=0.7)
    if ground_truth is not None:
        plt.plot(ground_truth, label='Ground Truth', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Line Plot of Predictions')
    plt.legend()
    line_plot_path = os.path.join(data_dir, 'line_plot.png')
    plt.savefig(line_plot_path)
    plt.close()
    return line_plot_path

