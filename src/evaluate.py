"""Утилиты оценки и интерпретации для churn-моделей."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _resolve_estimator(model: Any) -> Any:
    """Вернуть внутренний fitted estimator, если передан pipeline."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def get_prediction_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Вернуть positive-class scores для fitted classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))
    raise AttributeError("Model does not expose predict_proba or decision_function.")


def calculate_classification_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Посчитать основные classification-метрики на holdout set.

    Args:
        model: Обученный estimator с prediction interface.
        X_test: Holdout feature matrix.
        y_test: Holdout target vector.

    Returns:
        Словарь с accuracy, precision, recall, ROC-AUC и вспомогательными метриками.
    """
    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def calculate_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Посчитать classification-метрики для кастомного probability threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "targeted_customers": float(y_pred.sum()),
    }


def build_threshold_metrics_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Оценить precision/recall trade-offs на сетке threshold-значений."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)

    rows = [calculate_metrics_at_threshold(y_true, y_score, float(threshold)) for threshold in thresholds]
    threshold_table = pd.DataFrame(rows)
    threshold_table["f1"] = (
        2
        * threshold_table["precision"]
        * threshold_table["recall"]
        / (threshold_table["precision"] + threshold_table["recall"]).replace(0, np.nan)
    ).fillna(0.0)
    return threshold_table


def build_metrics_table(models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Оценить несколько моделей и вернуть сравнительную таблицу."""
    rows = []
    for model_name, model in models.items():
        metrics = calculate_classification_metrics(model, X_test, y_test)
        rows.append({"model": model_name, **metrics})

    metrics_table = pd.DataFrame(rows)
    return metrics_table[
        ["model", "accuracy", "precision", "recall", "roc_auc", "tn", "fp", "fn", "tp"]
    ].sort_values(["roc_auc", "recall", "precision"], ascending=[False, False, False]).reset_index(drop=True)


def get_confusion_matrix_frame(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Вернуть подписанную confusion matrix в виде DataFrame."""
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(
        matrix,
        index=["Actual: Stay", "Actual: Churn"],
        columns=["Predicted: Stay", "Predicted: Churn"],
    )


def get_roc_curve_data(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Вернуть точки ROC curve для fitted classifier."""
    y_score = get_prediction_scores(model, X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def find_optimal_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    strategy: str = "balanced",
) -> dict[str, float]:
    """Выбрать classification threshold на основе выбранной business strategy.

    Args:
        y_true: Истинные метки.
        y_score: Предсказанные churn probabilities.
        strategy: Стратегия выбора threshold.

    Returns:
        Словарь с выбранным threshold и вспомогательной статистикой.
    """
    threshold_table = build_threshold_metrics_table(y_true, y_score, thresholds=np.linspace(0.05, 0.95, 91))
    threshold_table["balanced_score"] = threshold_table["precision"] + threshold_table["recall"]

    if strategy == "recall":
        best_row = threshold_table.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    elif strategy == "precision":
        best_row = threshold_table.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]
    else:
        best_row = threshold_table.sort_values(["f1", "recall"], ascending=[False, False]).iloc[0]

    return best_row.to_dict()


def summarize_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Извлечь model-specific feature importance или коэффициенты.

    Args:
        model: Обученный estimator или pipeline.
        feature_names: Финальные имена признаков, используемых моделью.

    Returns:
        Отсортированную таблицу feature importance.
    """
    estimator = _resolve_estimator(model)

    if hasattr(estimator, "coef_"):
        coefficients = estimator.coef_.ravel()
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coefficients,
                "abs_importance": np.abs(coefficients),
            }
        )
        return importance_df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    if hasattr(estimator, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        )
        importance_df["abs_importance"] = importance_df["importance"].abs()
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    raise AttributeError("Estimator does not expose coefficients or feature importances.")


def aggregate_importance_by_feature_family(
    importance_df: pd.DataFrame,
    raw_feature_names: list[str],
) -> pd.DataFrame:
    """Агрегировать importance transformed features обратно к исходным feature families.

    Args:
        importance_df: Результат работы ``summarize_feature_importance``.
        raw_feature_names: Исходные имена признаков до one-hot encoding.

    Returns:
        DataFrame с агрегированной абсолютной важностью по feature families.
    """
    sorted_feature_names = sorted(raw_feature_names, key=len, reverse=True)

    def resolve_family(transformed_feature: str) -> str:
        for raw_feature in sorted_feature_names:
            if transformed_feature == raw_feature or transformed_feature.startswith(f"{raw_feature}_"):
                return raw_feature
        return transformed_feature

    aggregated = importance_df.copy()
    aggregated["feature_family"] = aggregated["feature"].map(resolve_family)
    family_summary = (
        aggregated.groupby("feature_family", as_index=False)
        .agg(
            total_abs_importance=("abs_importance", "sum"),
            mean_signed_importance=("importance", "mean"),
            transformed_features=("feature", "count"),
        )
        .sort_values("total_abs_importance", ascending=False)
        .reset_index(drop=True)
    )
    return family_summary


def assign_risk_segments(
    scores: pd.Series,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
) -> pd.Series:
    """Преобразовать churn probabilities в business-friendly risk segments.

    Args:
        scores: Предсказанные churn probabilities.
        low_threshold: Верхняя граница для low-risk customers.
        high_threshold: Нижняя граница для high-risk customers.

    Returns:
        Series с метками risk-сегментов.
    """
    bins = [-np.inf, low_threshold, high_threshold, np.inf]
    labels = ["Low risk", "Medium risk", "High risk"]
    return pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
