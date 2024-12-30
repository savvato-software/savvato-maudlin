import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def generate(config, data_dir, model, X_train, y_train, X_test, y_true, y_preds):

    # This generates the json and the image

    y_pred_probs = model.predict(X_test)
    
    threshold = config.get('prediction', {}).get('threshold', 0.5)

    y_preds = (y_pred_probs >= threshold).astype('int')

    # Compute TPR, FPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_preds)

    # Compute Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(data_dir, "roc_curve.png"))
    plt.close()

    print(f"ROC Curve report saved. firefox {data_dir}/roc_curve.png")
