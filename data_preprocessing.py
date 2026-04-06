"""Утилиты для загрузки, валидации и очистки churn-датасета."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_directory, log_step

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data(raw_data_path: str | Path) -> pd.DataFrame:
    """Загрузить raw churn-датасет с диска.

    Args:
        raw_data_path: Путь к CSV-файлу с raw IBM Telco dataset.

    Returns:
        Загруженный raw dataset в виде pandas DataFrame.
    """
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file was not found: {raw_data_path}")

    df = pd.read_csv(raw_data_path)
    object_columns = df.select_dtypes(include="object").columns
    if len(object_columns) > 0:
        df[object_columns] = df[object_columns].apply(lambda col: col.str.strip())
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Проверить, что raw dataset содержит ожидаемые колонки и типы.

    Args:
        df: Исходный dataset.
    """
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    unexpected_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    if unexpected_columns:
        raise ValueError(f"Unexpected columns found: {unexpected_columns}")

    if df["customerID"].duplicated().any():
        raise ValueError("customerID must be unique at the customer level.")

    if not {"Yes", "No"}.issuperset(set(df["Churn"].dropna().unique())):
        raise ValueError("Churn contains unexpected labels.")


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистить raw churn-датасет и подготовить его к feature engineering.

    Сюда входят преобразование типов, обработка пропусков и нормализация target.

    Args:
        df: Исходный dataset.

    Returns:
        Очищенный dataset, готовый для downstream-обработки.
    """
    validate_raw_schema(df)
    cleaned = df.copy()

    object_columns = cleaned.select_dtypes(include="object").columns
    if len(object_columns) > 0:
        cleaned[object_columns] = cleaned[object_columns].apply(lambda col: col.str.strip())

    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].map({1: "Yes", 0: "No"})
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"].replace("", pd.NA), errors="coerce")

    missing_total_charges = cleaned["TotalCharges"].isna()
    cleaned.loc[missing_total_charges & (cleaned["tenure"] == 0), "TotalCharges"] = 0.0

    if cleaned["TotalCharges"].isna().any():
        cleaned = cleaned.dropna(subset=["TotalCharges"]).copy()

    cleaned["ChurnFlag"] = cleaned["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return cleaned


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Сохранить processed dataset на диск.

    Args:
        df: Processed dataset для сохранения.
        output_path: Путь к итоговому файлу.
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)


def run_preprocessing(raw_data_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Выполнить preprocessing workflow целиком.

    Args:
        raw_data_path: Путь к raw dataset.
        output_path: Путь, куда нужно сохранить processed data.

    Returns:
        Финальный processed dataset.
    """
    log_step("Loading raw churn data.")
    df_raw = load_raw_data(raw_data_path)
    log_step("Cleaning churn data.")
    df_clean = clean_telco_data(df_raw)
    log_step("Saving processed churn data.")
    save_processed_data(df_clean, output_path)
    return df_clean
