
def generate(config, data_dir, model, X_train, y_train, X_test, y_true, y_preds):
    import shap
    import matplotlib.pyplot as plt

    # Generate SHAP values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Temporarily suppress the plot display
    plt.ioff()  # Turn off interactive mode

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False)  # Prevent auto-display

    # Save the plot to a file
    plt.savefig(f"{data_dir}/shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print(f"SHAP summary saved. firefox {data_dir}/shap_summary_plot.png")

