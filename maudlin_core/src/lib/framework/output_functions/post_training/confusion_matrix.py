import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


def generate(config, data_dir, model,  X_train, y_train, X_test, y_true, y_preds):

    # Predict using the model
    y_pred = model.predict(X_test)

    # Determine task type and process predictions
    task_type = config.get("task_type", "binary")
    if task_type == "binary":
        y_pred_labels = (y_pred > 0.5).astype(int)
        y_true_labels = y_true.astype(int)
    elif task_type == "multi-class":
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    print("Confusion Matrix:")
    print(cm)
    # Display and save the confusion matrix
    class_labels = config.get("class_labels", None)
    if class_labels:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(data_dir, "confusion_matrix.png"))
    plt.close()

    print(f"Confusion Matrix saved to {data_dir}/confusion_matrix.png")

    # Generate and save classification metrics
    print("\nClassification Metrics:")
    report = classification_report(y_true_labels, y_pred_labels, target_names=class_labels, output_dict=True)
    print(json.dumps(report, indent=4))

    with open(os.path.join(data_dir, "classification_report.json"), "w") as f:
        f.write(json.dumps(classification_report(y_true_labels, y_pred_labels, target_names=class_labels, output_dict=True), indent=4))

    with open(os.path.join(data_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true_labels, y_pred_labels, target_names=class_labels, digits=4))

    print(f"Classification report saved to {data_dir}/classification_report.txt")


