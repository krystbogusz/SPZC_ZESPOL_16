import datetime

import numpy as np
import pandas as pd

from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    drop_cols = ["url", "type"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["type"]

    return X, y


def compute_class_weigths(y_train) -> dict[Any, Any]:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    return class_weights


def split_dataset(X, y) -> tuple[Any, Any, Any, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_test, X_train, y_test, y_train


def get_timestamp() -> str:
    ct = datetime.datetime.now()

    return ct.strftime("%Y-%m-%d_%H-%M-%S")
