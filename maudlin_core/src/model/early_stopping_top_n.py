import keras
import numpy as np
import optuna

class EarlyStoppingTopN(keras.callbacks.Callback):

    def __init__(self, metric_name, trial, best_trials, patience=5, delta=0.001, top_n=3, patience_credit=0.25, strike_tolerance=0.1, trend_window=4, trend_boost=0.6, lbpr_start_count=2):
        """
        Early stopping callback that compares the current run's performance against the best N runs.
        Args:
        - metric_name (str): Metric to monitor (e.g., 'val_auc').
        - trial (optuna.Trial): Current Optuna trial for pruning.
        - best_trials (list): List of top N trials with historical metrics.
        - patience (int): Number of epochs to wait for improvement.
        - delta (float): Minimum improvement threshold.
        - top_n (int): Number of top trials to compare against.
        - patience_credit (float): Additional patience added for each improvement.
        - strike_tolerance (float): Percentage drop threshold for strikes.
        - trend_window (int): Number of epochs to calculate improvement trend.
        - trend_boost (int): Additional patience if a positive trend is detected.
        - lbpr_start_count (int): Starting count for Last Best Performing Run comparison.
        """
        super().__init__()
        self.monitor = metric_name
        self.trial = trial
        self.best_trials = best_trials
        self.patience = int(patience)
        self.delta = float(delta)
        self.top_n = int(top_n)
        self.patience_credit = float(patience_credit)
        self.strike_tolerance = float(strike_tolerance)

        # Trend detection
        self.trend_window = trend_window
        self.trend_boost = trend_boost

        # LBPR counter
        self.lbpr_counter = lbpr_start_count

        # Internal state
        self.wait = 0
        self.dynamic_patience = self.patience
        self.best_value = float('-inf') if "auc" in metric_name else float('inf')
        self.history = []
        self.strikes = 0

    def detect_trend(self):
        """Detects a positive trend over the last N values."""
        if len(self.history) < self.trend_window:
            return False  # Not enough data to calculate trend

        recent = self.history[-self.trend_window:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]  # Slope of trend line
        return trend > 0  # Positive slope indicates improvement

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)

        if current_value is None:
            print(f"Warning: Metric {self.monitor} is not available in logs.")
            return

        # Record metric history
        self.history.append(current_value)

        # LBPR (Last Best Performing Run) check
        worst_top = self.best_trials[-1] if self.best_trials else {'values': []}
        worst_top_value = worst_top['values'][epoch] if epoch < len(worst_top['values']) else -np.inf

        if current_value > worst_top_value:
            self.lbpr_counter += 1
        else:
            self.lbpr_counter -= 1

        if self.lbpr_counter < 0:
            print(f"\n\nTrial was not performing as well as the worst-performing best run. Pruning trial {self.trial.number}.")
            self.model.stop_training = True
            self.trial.report(current_value, epoch)
            raise optuna.TrialPruned()

        # Check if current run is competitive with top N runs
        if len(self.best_trials) >= self.top_n:
            if current_value < worst_top_value * (1 - self.strike_tolerance):
                self.strikes += 1
                print(f" 🙅🏾 {self.strikes}. Current: {current_value:.4f}, Worst Top-N: {worst_top_value:.4f}.")
                if self.strikes >= self.dynamic_patience:
                    print(f" 🙅🏾 Too many strikes. Pruning trial {self.trial.number}.")
                    self.model.stop_training = True
                    self.trial.report(current_value, epoch)
                    raise optuna.TrialPruned()

        # Check patience for stopping within the current trial
        if current_value > self.best_value + self.delta:
            self.best_value = current_value
            self.wait = 0
            self.dynamic_patience += self.patience_credit
            print(f" ⬆️ P: {self.dynamic_patience:.2f}")
        else:
            self.wait += 1
            print(f" 🐢 # {self.wait} / {self.dynamic_patience:.2f}")
            if self.wait >= self.dynamic_patience:
                if not self.detect_trend():
                    print(f"Epoch {epoch + 1}: Patience exceeded. Stopping trial {self.trial.number}.")
                    self.model.stop_training = True
                    self.trial.report(current_value, epoch)
                    raise optuna.TrialPruned()
                else:
                    self.dynamic_patience += self.trend_boost
                    print(f"Positive trend detected! Boosting patience to {self.dynamic_patience}.")

    def on_train_end(self, logs=None):
        # Save the current trial's results
        current_trial = {
            'values': self.history,
            'trial': self.trial.number,
            'params': self.trial.params,
        }

        # Add the current trial to the list and sort by performance
        self.best_trials.append(current_trial)
        self.best_trials.sort(key=lambda x: x['values'][-1], reverse=True)

        # Trim to keep only the top N trials
        del self.best_trials[self.top_n:]

        # Check if the current trial is still in the top N
        if any(t['trial'] == self.trial.number for t in self.best_trials):
            position = next(i for i, t in enumerate(self.best_trials) if t['trial'] == self.trial.number)
            print(f"\nTrial {self.trial.number} added to best_trials at position {position + 1}.")
        else:
            print(f"\nTrial {self.trial.number} NOT added to best_trials.")
