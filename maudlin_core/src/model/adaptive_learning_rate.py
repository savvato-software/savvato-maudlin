import keras

class AdaptiveLearningRate(keras.callbacks.Callback):

    def __init__(self, metric_name, patience=5, factor=0.5, min_lr=1e-6, reduction_grace_period=0):
        """
        Adjust learning rate dynamically when monitored metric stops improving.
        Args:
        - monitor (str): Metric to monitor (e.g., 'val_loss', 'val_auc').
        - patience (int): Number of epochs to wait before reducing the learning rate.
        - factor (float): Factor by which to reduce the learning rate (e.g., 0.5 for halving).
        - min_lr (float): Minimum allowable learning rate.
        """
        super().__init__()
        self.monitor = metric_name 
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.orig_reduction_grace_period = int(reduction_grace_period)
        self.reduction_grace_period = int(reduction_grace_period)
        self.wait = 0
        self.best_value = float('inf') if "loss" in metric_name else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            print(f"Warning: Metric {self.monitor} is not available in logs.")
            return

        # Check if the monitored metric has improved
        if ("loss" in self.monitor and current_value < self.best_value) or \
           ("loss" not in self.monitor and current_value > self.best_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.reduction_grace_period > 0:
                self.reduction_grace_period -= 1
            elif self.wait >= self.patience:
                self._reduce_lr(epoch)
                self.patience += 1
                self.reduction_grace_period = self.orig_reduction_grace_period

    def _reduce_lr(self, epoch):
        optimizer = self.model.optimizer

        # Safely get the current learning rate
        try:
            old_lr = float(optimizer.learning_rate.numpy())
        except AttributeError:
            old_lr = float(optimizer.lr.numpy())  # For older TensorFlow versions

        # Compute the new learning rate
        new_lr = max(old_lr * self.factor, self.min_lr)

        # Safely set the new learning rate
        try:
            optimizer.learning_rate.assign(new_lr)
        except AttributeError:
            keras.backend.set_value(optimizer.learning_rate, new_lr)  # Fallback

        print(f"\033[35Epoch {epoch + 1}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}.")

