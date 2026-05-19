import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from config import FINAL_DATASET, RUNS_DIR, N_REPEATS
from evaluate import evaluate, get_predictions
from plots import (
    plot_confusion_matrix,
    plot_feature_importance_using_permutation_importance,
)
from training_utils import (
    load_data,
    split_dataset,
    get_timestamp,
)

def run_nb():
    main()

def build_nb_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("nb", GaussianNB(var_smoothing=1e-8)),
        ]
    )


def train_nb(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_test, X_train, y_test, y_train = split_dataset(X, y_enc)

    pipeline = build_nb_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, X_train, y_test, y_train, le


def main():
    model_name = "Naive Bayes"

    current_timestamp = get_timestamp()
    current_runs_dir = RUNS_DIR / f"nb_run_{current_timestamp}"
    current_runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Runs directory: {current_runs_dir}")

    X, y = load_data(FINAL_DATASET)
    feature_names = list(X.columns)

    print(f"Loaded {len(X):,} rows | Features: {X.shape[1]}")
    print(f"Training {model_name}...")

    pipeline, X_test, X_train, y_test, y_train, le = train_nb(X, y)

    print(f"Evaluating {model_name}...")

    y_pred = get_predictions(pipeline, X_test)
    evaluate(y_test, y_pred, current_runs_dir, model_name)

    print(f"Plotting {model_name}...")

    plot_confusion_matrix(y_test, y_pred, current_runs_dir, model_name, le)

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