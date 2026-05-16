import json
from pathlib import Path

from sklearn.metrics import classification_report


def evaluate(y_test, y_pred, output_dir: Path):
    get_classification_report(y_test, y_pred, output_dir)


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
