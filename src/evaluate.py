from sklearn.metrics import classification_report


def evaluate(y_test, y_pred):
    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_test, y_pred, digits=4))


def get_predictions(pipeline, X_test):
    return pipeline.predict(X_test)
