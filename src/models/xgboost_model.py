import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import RUNS_DIR, FINAL_DATASET, N_REPEATS
from evaluate import get_predictions, evaluate
from plots import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance_using_permutation_importance,
)
from training_utils import (
    split_dataset,
    compute_class_weigths,
    get_timestamp,
    load_data,
)

def run_xgboost():
    main()

def build_xgboost_pipeline(class_weights_str: dict, le: LabelEncoder) -> Pipeline:
    xgb_weights = {i: class_weights_str[cls] for i, cls in enumerate(le.classes_)}

    return Pipeline(
        [
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def train_xgboost(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_test, X_train, y_test, y_train = split_dataset(X, y_enc)

    class_weights_int = compute_class_weigths(y_train)
    class_weights_str = {le.classes_[k]: v for k, v in class_weights_int.items()}
    sample_weight_train = np.array([class_weights_int[c] for c in y_train])

    pipeline = build_xgboost_pipeline(class_weights_str, le)
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight_train)

    return pipeline, X_test, X_train, y_test, y_train, le


def main():
    model_name = "XGBoost"

    current_timestamp = get_timestamp()
    current_runs_dir = RUNS_DIR / f"xgboost_run_{current_timestamp}"
    current_runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Runs directory: {current_runs_dir}")

    X, y = load_data(FINAL_DATASET)
    feature_names = list(X.columns)

    print(f"Loaded {len(X):,} rows | Features: {X.shape[1]}")
    print(f"Training {model_name}...")

    pipeline, X_test, X_train, y_test, y_train, le = train_xgboost(X, y)

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
        n_repeats=N_REPEATS,
    )

    print("Done!")


if __name__ == "__main__":
    main()
