import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score


def plot_confusion_matrix(y_test, y_pred, output_dir: Path, model_name: str, le):
    class_names = list(le.classes_)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(pipeline, feature_names, output_dir: Path, model_name: str, top_n: int = 10):
    model = pipeline[-1] if hasattr(pipeline, "steps") else pipeline

    if hasattr(model, "coef_"):
        importances = np.atleast_2d(model.coef_)

        if importances.shape[0] > 1 and hasattr(model, "classes_"):
            titles = [f"{model_name}\nclass: {cls}" for cls in model.classes_]
        else:
            titles = [model_name]

        cmap_color = "steelblue"
        show_zero_line = True
    elif hasattr(model, "feature_importances_"):
        importances = np.atleast_2d(model.feature_importances_)
        titles = [model_name]
        cmap_color = "steelblue"
        show_zero_line = False
    else:
        raise AttributeError(
            f"Model '{model_name}' does not expose coef_ or feature_importances_. Cannot plot feature importances."
        )

    n_plots = len(importances)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(5 * n_plots, 5),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, title, imp in zip(axes, titles, importances):
        imp_array = np.asarray(imp).flatten()

        idx = np.argsort(np.abs(imp_array))[-top_n:]

        colors = ["#C44E52" if v < 0 else cmap_color for v in imp_array[idx]]

        ax.barh(
            np.array(feature_names)[idx],
            imp_array[idx],
            color=colors,
        )

        if show_zero_line:
            ax.axvline(0, color="black", linewidth=0.8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Importance / Coefficient")

    fig.suptitle(f"Top {top_n} Features — {model_name}", fontsize=13, y=1.02)
    plt.tight_layout()

    filename = f"{model_name.replace(' ', '_').lower()}_feature_importance.png"

    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_using_permutation_importance(
    pipeline,
    feature_names,
    X_test,
    y_test,
    le: LabelEncoder,
    output_dir: Path,
    model_name: str,
    top_n: int = 10,
    n_repeats: int = 10,
    random_state: int = 42,
) -> None:

    classes = le.classes_
    n_classes = len(classes)
    feature_names = np.asarray(feature_names)

    all_importances = []

    for class_idx, class_name in enumerate(classes):

        def _binary_f1(estimator, X, y, _idx=class_idx):
            y_pred = estimator.predict(X)
            return f1_score(y == _idx, y_pred == _idx, average="binary")

        scorer = _binary_f1

        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring=scorer,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        all_importances.append(result.importances_mean)

    all_importances = np.array(all_importances)

    abs_max = np.abs(all_importances).max()
    x_lim = (-abs_max * 0.15, abs_max * 1.15)

    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(5 * n_classes, 5),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, class_name, imp in zip(axes, classes, all_importances):
        idx = np.argsort(np.abs(imp))[-top_n:]
        values = imp[idx]
        labels = feature_names[idx]

        colors = ["#C44E52" if v < 0 else "steelblue" for v in values]

        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(x_lim)
        ax.set_title(f"class: {class_name}", fontsize=10)
        ax.set_xlabel("Mean F1 drop (permutation)")

    fig.suptitle(
        f"Top {top_n} Features per Class — {model_name}",
        fontsize=12,
        y=1.02,
    )

    filename = f"{model_name.replace(' ', '_').lower()}_feature_importance_using_permutation_importance.png"

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
