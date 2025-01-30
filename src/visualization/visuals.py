import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve


def save_confusion_matrix(y_true, y_pred, plots_dir):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()


def save_roc_curve(y_true, y_proba, roc_auc, plots_dir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()


def save_histogram(y_true, y_pred, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(y_true, bins=20, alpha=0.5, label="True Labels", color="blue")
    plt.hist(y_pred, bins=20, alpha=0.5, label="Predicted Labels", color="red")
    plt.xlabel("Churn")
    plt.ylabel("Frequency")
    plt.title("Histogram of True Labels vs. Predicted Labels")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "histogram_true_vs_predicted_labels.png"))
    plt.close()


def save_density_plot(y_true, y_pred, plots_dir):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true, fill=True, label="True Labels", color="blue")
    sns.kdeplot(y_pred, fill=True, label="Predicted Labels", color="red")
    plt.xlabel("Churn")
    plt.ylabel("Density")
    plt.title("Density Plot of True Labels vs. Predicted Labels")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "density_true_vs_predicted_labels.png"))
    plt.close()


def save_feature_importance(model, feature_names, plots_dir):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    )
    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False
    )

    plt.figure(figsize=(18, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance,
        hue="feature",
        palette="deep",
        legend=False,
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
    plt.close()
