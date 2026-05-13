from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import FINAL_DATASET, RUNS_DIR
from evaluate import evaluate, get_predictions
from plots import plot_confusion_matrix, plot_feature_importance
from training_utils import load_data, compute_class_weigths, split_dataset, get_timestamp


def build_rf_pipeline(class_weights: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )),
    ])


def train_rf(X, y):
    X_test, X_train, y_test, y_train = split_dataset(X, y)
    class_weights = compute_class_weigths(y_train)

    pipeline = build_rf_pipeline(class_weights)
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, X_train, y_test, y_train


def main():
    current_timestamp = get_timestamp()
    current_runs_dir = RUNS_DIR / f"rf_run_{current_timestamp}"
    current_runs_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(FINAL_DATASET)
    feature_names = list(X.columns)

    pipeline, X_test, X_train, y_test, y_train = train_rf(X, y)
    y_pred = get_predictions(pipeline, X_test)
    evaluate(y_test, y_pred)

    plot_confusion_matrix(y_test, y_pred, labels=pipeline.named_steps["rf"].classes_, output_dir=current_runs_dir, model_name='Random Forest')
    plot_feature_importance(pipeline, feature_names, output_dir=current_runs_dir, top_n=10, model_name='Random Forest')


if __name__ == '__main__':
    main()