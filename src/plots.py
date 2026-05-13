from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_test, y_pred, labels, output_dir: Path):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")

    ax.set_title("Confusion Matrix — URL SVM Classifier", fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)


def plot_feature_importance(pipeline, feature_names, output_dir: Path, top_n: int = 10):
    coefs = pipeline.named_steps["svm"].coef_
    classes = pipeline.named_steps["svm"].classes_

    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 5), sharey=False)

    for ax, cls, coef in zip(axes, classes, coefs):
        idx = np.argsort(np.abs(coef))[-top_n:]
        ax.barh(np.array(feature_names)[idx], coef[idx], color="steelblue")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cls, fontsize=11)
        ax.set_xlabel("Coefficient")

    fig.suptitle(f"Top {top_n} Features per Class (LinearSVC)", fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
