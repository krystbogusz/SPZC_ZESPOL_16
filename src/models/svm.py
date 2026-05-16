import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from config import FINAL_DATASET, RUNS_DIR
from evaluate import evaluate, get_predictions
from plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_feature_importance_using_permutation_importance,
)
from training_utils import (
    load_data,
    compute_class_weigths,
    split_dataset,
    get_timestamp,
)


def build_svm_pipeline(class_weights: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    max_iter=2000,
                    class_weight=class_weights,
                    dual="auto",
                    random_state=42,
                ),
            ),
        ]
    )


def train_svm(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_test, X_train, y_test, y_train = split_dataset(X, y_enc)

    class_weights = compute_class_weigths(y_train)

    pipeline = build_svm_pipeline(class_weights)
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, X_train, y_test, y_train, le


def main():
    model_name = "SVM (LinearSVC)"

    current_timestamp = get_timestamp()
    current_runs_dir = RUNS_DIR / f"svm_run_{current_timestamp}"
    current_runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Runs directory: {current_runs_dir}")

    X, y = load_data(FINAL_DATASET)
    feature_names = list(X.columns)

    print(f"Loaded {len(X):,} rows | Features: {X.shape[1]}")
    print(f"Training {model_name}...")

    pipeline, X_test, X_train, y_test, y_train, le = train_svm(X, y)

    print(f"Evaluating {model_name}...")

    y_pred = get_predictions(pipeline, X_test)
    evaluate(y_test, y_pred, current_runs_dir, model_name)

    print(f"Plotting {model_name}...")

    plot_confusion_matrix(y_test, y_pred, current_runs_dir, model_name, le)
    plot_feature_importance(pipeline, feature_names, current_runs_dir, model_name)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), 5_000, replace=False)

    plot_feature_importance_using_permutation_importance(
        pipeline,
        feature_names,
        X_test.iloc[idx],
        y_test[idx],
        le,
        current_runs_dir,
        model_name,
        n_repeats=5,
    )

    print("Done!")


if __name__ == "__main__":
    main()
