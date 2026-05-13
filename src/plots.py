from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_test, y_pred, output_dir: Path, model_name: str, labels=None
):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(
    pipeline, feature_names, output_dir: Path, model_name: str, top_n: int = 10
):
    model = pipeline[-1] if hasattr(pipeline, "steps") else pipeline

    if hasattr(model, "coef_"):
        importances = np.atleast_2d(model.coef_)
        if len(importances) > 1 and hasattr(model, "classes_"):
            titles = [str(cls) for cls in model.classes_]
        else:
            titles = [model_name]
    elif hasattr(model, "feature_importances_"):
        importances = np.atleast_2d(model.feature_importances_)
        titles = [model_name]
    else:
        raise AttributeError(
            f"Model '{model_name}' does not support feature importances extraction."
        )

    fig, axes = plt.subplots(
        1, len(importances), figsize=(5 * len(importances), 5), squeeze=False
    )
    axes = axes.flatten()

    for ax, title, imp in zip(axes, titles, importances):
        imp_array = np.asarray(imp).flatten()
        idx = np.argsort(np.abs(imp_array))[-top_n:]

        ax.barh(np.array(feature_names)[idx], imp_array[idx], color="steelblue")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Importance / Coefficient")

    fig.suptitle(f"Top {top_n} Features ({model_name})", fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)