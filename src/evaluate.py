import json
from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, balanced_accuracy_score, matthews_corrcoef


def evaluate(y_test, y_pred, output_dir: Path, model_name: str):
    get_classification_metrics(y_pred, y_test, output_dir)
    get_classification_report(y_test, y_pred, output_dir)


def get_classification_metrics(y_pred, y_test, output_dir: Path) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_micro": precision_score(y_test, y_pred, average="micro"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_micro": recall_score(y_test, y_pred, average="micro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    json.dump(metrics, open(output_dir / "metrics.json", "w"), indent=4)

    return metrics


def get_classification_report(y_test, y_pred, output_dir: Path):
    cr = classification_report(y_test, y_pred, digits=4)

    print("\n── Classification Report ──────────────────────────────")
    print(cr)

    report_path = output_dir / "classification_report.txt"

    if isinstance(cr, dict):
        report_str = json.dumps(cr, indent=2)
    else:
        report_str = cr

    with open(report_path, "w") as f:
        f.write(report_str)


def get_predictions(pipeline, X_test):
    return pipeline.predict(X_test)
