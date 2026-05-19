import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

from config import RUNS_DIR, FINAL_DATASET, N_REPEATS
from evaluate import evaluate, get_predictions
from plots import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance_using_permutation_importance,
)
from training_utils import (
    get_timestamp,
    load_data,
    split_dataset,
    compute_class_weigths,
)
from timer import Timer

def run_lr():
    main()


def build_lr_pipeline(class_weights: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    max_iter=3000,
                    class_weight=class_weights,
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def train_lr(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_test, X_train, y_test, y_train = split_dataset(X, y_enc)

    class_weights = compute_class_weigths(y_train)

    pipeline = build_lr_pipeline(class_weights)

    timer = Timer(logger=None)
    timer.start()
    pipeline.fit(X_train, y_train)
    train_time = timer.stop()

    return pipeline, X_test, X_train, y_test, y_train, le, train_time


def main():
    model_name = "Logistic Regression"

    current_timestamp = get_timestamp()
    current_runs_dir = RUNS_DIR / f"lr_run_{current_timestamp}"
    current_runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Runs directory: {current_runs_dir}")

    X, y = load_data(FINAL_DATASET)
    feature_names = list(X.columns)

    print(f"Loaded {len(X):,} rows | Features: {X.shape[1]}")
    print(f"Training {model_name}...")

    pipeline, X_test, X_train, y_test, y_train, le, train_time = train_lr(X, y)
    print(f"Training time: {train_time:.4f} seconds")

    print(f"Evaluating {model_name}...")

    timer = Timer(logger=None)
    timer.start()
    y_pred = get_predictions(pipeline, X_test)
    pred_time = timer.stop()
    print(f"Prediction time: {pred_time:.4f} seconds")

    evaluate(y_test, y_pred, current_runs_dir, model_name, train_time, pred_time)

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
        n_repeats=N_REPEATS,
    )

    print("Done!")


if __name__ == "__main__":
    main()
