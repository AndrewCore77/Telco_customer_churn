"""Утилиты подготовки признаков для churn-моделирования."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def prepare_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Применить легкий feature engineering перед train/test split.

    Engineered features намеренно сделаны простыми и понятными для бизнеса:
    они добавляют контекст customer lifecycle и глубины использования продукта,
    не опираясь на target variable.

    Args:
        df: Очищенный dataset.

    Returns:
        Dataset, готовый к подготовке для моделирования.
    """
    modeling_df = df.copy()

    required_columns = {
        "tenure",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        *SERVICE_COLUMNS,
    }
    missing_columns = sorted(required_columns.difference(modeling_df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns required for feature engineering: {missing_columns}")

    modeling_df["tenure_group"] = pd.cut(
        modeling_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    modeling_df["is_new_customer"] = np.where(modeling_df["tenure"] <= 12, "Yes", "No")

    service_flags = [
        (modeling_df["PhoneService"] == "Yes").astype(int),
        (modeling_df["MultipleLines"] == "Yes").astype(int),
        (modeling_df["InternetService"] != "No").astype(int),
        (modeling_df["OnlineSecurity"] == "Yes").astype(int),
        (modeling_df["OnlineBackup"] == "Yes").astype(int),
        (modeling_df["DeviceProtection"] == "Yes").astype(int),
        (modeling_df["TechSupport"] == "Yes").astype(int),
        (modeling_df["StreamingTV"] == "Yes").astype(int),
        (modeling_df["StreamingMovies"] == "Yes").astype(int),
    ]
    modeling_df["num_services"] = sum(service_flags)

    modeling_df["avg_monthly_spend_proxy"] = np.where(
        modeling_df["tenure"] > 0,
        modeling_df["TotalCharges"] / modeling_df["tenure"],
        modeling_df["MonthlyCharges"],
    )
    modeling_df["has_auto_payment"] = np.where(
        modeling_df["PaymentMethod"].str.contains("automatic", case=False, na=False),
        "Yes",
        "No",
    )

    return modeling_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "ChurnFlag",
    drop_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Разделить dataset на feature matrix и target vector.

    Args:
        df: Входной dataset, содержащий признаки и target.
        target_column: Название target-колонки.
        drop_columns: Дополнительные колонки, которые нужно исключить из X.

    Returns:
        Кортеж из feature matrix и target vector.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    y = df[target_column].copy()
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})
    y = y.astype(int)

    columns_to_drop = {target_column, "customerID"}
    if target_column != "Churn" and "Churn" in df.columns:
        columns_to_drop.add("Churn")
    if target_column != "ChurnFlag" and "ChurnFlag" in df.columns:
        columns_to_drop.add("ChurnFlag")
    if drop_columns is not None:
        columns_to_drop.update(drop_columns)

    X = df.drop(columns=[column for column in columns_to_drop if column in df.columns]).copy()
    return X, y


def infer_feature_groups(
    df: pd.DataFrame,
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Определить группы numeric и categorical признаков.

    Args:
        df: Входная feature matrix.
        numeric_features: Необязательный явный список numeric-признаков.
        categorical_features: Необязательный явный список categorical-признаков.

    Returns:
        Словарь со сгруппированными именами признаков.
    """
    if numeric_features is None:
        inferred_numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    else:
        inferred_numeric = [feature for feature in numeric_features if feature in df.columns]

    if categorical_features is None:
        inferred_categorical = [column for column in df.columns if column not in inferred_numeric]
    else:
        inferred_categorical = [feature for feature in categorical_features if feature in df.columns]

    return {
        "numeric": inferred_numeric,
        "categorical": inferred_categorical,
    }


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Создать воспроизводимый train/test split со стратификацией по target.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Доля строк, зарезервированная под test evaluation.
        random_state: Seed для воспроизводимости.

    Returns:
        Train/test split для признаков и target.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    feature_groups: dict[str, list[str]],
    scale_numeric: bool = True,
    one_hot_drop: str | None = "if_binary",
) -> ColumnTransformer:
    """Собрать preprocessing transformer для modeling pipelines.

    Numeric features при необходимости масштабируются. Categorical features
    проходят импутацию и one-hot encoding с обработкой неизвестных категорий
    для безопасного inference.

    Args:
        feature_groups: Mapping с именами numeric и categorical признаков.
        scale_numeric: Нужно ли стандартизировать numeric-признаки.
        one_hot_drop: Политика drop для one-hot encoding. ``if_binary`` —
            практичный выбор для линейных моделей, потому что он сокращает
            избыточные бинарные колонки и сохраняет мультиклассовые категории.

    Returns:
        Настроенный column transformer.
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=one_hot_drop,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_groups.get("numeric", [])),
            ("cat", categorical_transformer, feature_groups.get("categorical", [])),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Вернуть имена выходных признаков из fitted preprocessor."""
    return preprocessor.get_feature_names_out().tolist()
