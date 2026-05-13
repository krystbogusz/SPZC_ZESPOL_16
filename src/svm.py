from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import FINAL_DATASET, RUNS_DIR
from evaluate import evaluate, get_predictions
from plots import plot_confusion_matrix, plot_feature_importance
from prepare_data_for_training import load_data, compute_class_weigths, split_dataset, get_timestamp


def build_svm_pipeline(class_weights: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight=class_weights,
            dual="auto",
            random_state=42,
        )),
    ])


def train_svm(X, y):
    X_test, X_train, y_test, y_train = split_dataset(X, y)
    class_weights = compute_class_weigths(y_train)

    pipeline = build_svm_pipeline(class_weights)
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, X_train, y_test, y_train


def main():
    current_timestamp = get_timestamp()
    current_runs_dir = RUNS_DIR / f"svm_run_{current_timestamp}"
    current_runs_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(FINAL_DATASET)
    feature_names = list(X.columns)

    pipeline, X_test, X_train, y_test, y_train = train_svm(X, y)
    y_pred = get_predictions(pipeline, X_test)
    evaluate(y_test, y_pred)

    plot_confusion_matrix(y_test, y_pred, labels=pipeline.named_steps["svm"].classes_, output_dir=current_runs_dir)
    plot_feature_importance(pipeline, feature_names, output_dir=current_runs_dir, top_n=10)


if __name__ == '__main__':
    main()
