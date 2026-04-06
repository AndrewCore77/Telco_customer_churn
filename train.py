"""Утилиты обучения моделей для churn prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_model() -> BaseEstimator:
    """Создать простой baseline classifier для сравнения.

    ``prior`` — это сильный baseline для imbalanced classification:
    предсказанный класс остается классом большинства, но probability outputs
    отражают наблюдаемое распределение классов и дают осмысленный ROC-AUC baseline.
    """
    return DummyClassifier(strategy="prior")


def build_logistic_regression_model(preprocessor: Any) -> BaseEstimator:
    """Создать pipeline для logistic regression с preprocessing.

    Args:
        preprocessor: Transformer, который готовит признаки для линейного моделирования.

    Returns:
        Настроенный pipeline для logistic regression.
    """
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    max_iter=2_000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_random_forest_model(preprocessor: Any) -> BaseEstimator:
    """Создать pipeline для random forest с preprocessing.

    Args:
        preprocessor: Transformer, который готовит признаки для tree-based modeling.

    Returns:
        Настроенный pipeline для random forest.
    """
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """Обучить модель на training data.

    Args:
        model: Необученный estimator или pipeline.
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Обученная модель.
    """
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
) -> dict[str, BaseEstimator]:
    """Обучить полный набор candidate models.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        preprocessor: Либо общий preprocessor, либо mapping с ключами
            ``logistic_regression`` and ``random_forest``.

    Returns:
        Словарь, который отображает названия моделей на обученные estimators.
    """
    if isinstance(preprocessor, dict):
        logistic_preprocessor = preprocessor["logistic_regression"]
        random_forest_preprocessor = preprocessor.get("random_forest", logistic_preprocessor)
    else:
        logistic_preprocessor = preprocessor
        random_forest_preprocessor = preprocessor

    models = {
        "Baseline": build_baseline_model(),
        "Logistic Regression": build_logistic_regression_model(logistic_preprocessor),
        "Random Forest": build_random_forest_model(random_forest_preprocessor),
    }

    return {
        model_name: train_model(model, X_train, y_train)
        for model_name, model in models.items()
    }
