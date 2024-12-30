import keras
import os
import json
from termcolor import colored


class TrackBestMetric(keras.callbacks.Callback):
    # Define metrics where lower values are better
    LOWER_IS_BETTER = {'mae': True}

    def __init__(self, metric_names, log_dir):
        super().__init__()
        self.metric_names = metric_names  # List of metrics to track
        self.best_values = {metric: float('inf') if self.LOWER_IS_BETTER.get(metric, False) else float('-inf') for metric in metric_names}
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            updated_metrics = {}
            line_output = f"Epoch {epoch + 1}: "

            for metric in self.metric_names:
                if metric in logs:
                    current_value = logs[metric]
                    lower_is_better = self.LOWER_IS_BETTER.get(metric, False)

                    if lower_is_better:
                        # Handle metrics where lower is better
                        if current_value < self.best_values[metric]:
                            self.best_values[metric] = current_value
                            value_str = f"{metric}: {current_value:.4f} (Best so far)"
                        else:
                            value_str = colored(f"{metric}: {current_value:.4f}", 'blue')
                    else:
                        # Handle metrics where higher is better
                        if current_value > self.best_values[metric]:
                            self.best_values[metric] = current_value
                            value_str = f"{metric}: {current_value:.4f} (Best so far)"
                        else:
                            value_str = colored(f"{metric}: {current_value:.4f}", 'blue')

                    updated_metrics[metric] = {"epoch": epoch + 1, "best_value": self.best_values[metric]}
                    line_output += value_str + ' | '

            # Save the best metrics to a single file
            with open(os.path.join(self.log_dir, "best_metrics.json"), "w") as f:
                json.dump(updated_metrics, f, indent=4)

            print(line_output.strip(' | '))

    def on_train_end(self, logs=None):
        print("Training complete. Best metrics:")
        for metric, value in self.best_values.items():
            print(f"{metric} = {value:.4f}")

