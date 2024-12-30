import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def generate(predictions, data_dir, ground_truth):
    prob_true, prob_pred = calibration_curve(ground_truth, predictions, n_bins=10, strategy='uniform')
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    calibration_curve_path = os.path.join(data_dir, "calibration_curve.png")
    plt.savefig(calibration_curve_path)
    plt.close()
    print(f"Calibration curve saved to {calibration_curve_path}")
    return calibration_curve_path



