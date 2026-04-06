"""Общие utility-функции для churn prediction проекта."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Вернуть корневую директорию проекта."""
    return Path(__file__).resolve().parent.parent


def ensure_directory(path: str | Path) -> Path:
    """Создать директорию, если она еще не существует.

    Args:
        path: Путь к директории, которую нужно создать.

    Returns:
        Нормализованный объект пути.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    """Сохранить JSON-сериализуемый словарь на диск.

    Args:
        payload: Сериализуемый словарь с метаданными или метриками.
        output_path: Путь к итоговому файлу.
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log_step(message: str) -> None:
    """Обеспечить легкий project logging для скриптов и ноутбуков.

    Args:
        message: Человекочитаемое сообщение о прогрессе.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
