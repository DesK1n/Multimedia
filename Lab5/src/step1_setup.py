"""Крок 1. Налаштування середовища та завантаження базової моделі YOLOv8.

Перевіряє доступність фреймворку Ultralytics, завантажує попередньо навчені
ваги YOLOv8n у локальний каталог та зберігає метадані класів COCO.
"""

import json
from pathlib import Path
from typing import Any, Dict

import ultralytics
from ultralytics import YOLO

from src import config, utils


def run_setup() -> Dict[str, Any]:
    """Виконує ініціалізацію та перевірку робочого середовища.

    Returns:
        Dict[str, Any]: Словник із метаданими налаштованого середовища.
    """
    print(f"Версія Ultralytics: {ultralytics.__version__}")

    # Завантажуємо ваги моделі YOLOv8n у каталог weights
    model_path = config.YOLO_NANO_WEIGHTS
    model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
    if not model_path.exists():
        # Зберігаємо завантажені ваги у наш локальний каталог weights
        import shutil

        default_downloaded = Path("yolov8n.pt")
        if default_downloaded.exists():
            shutil.move(str(default_downloaded), str(model_path))

    print(f"Модель YOLOv8n готова: {model_path}")

    class_names = list(model.names.values())
    print(f"Кількість класів COCO: {len(class_names)}")
    print(f"Приклади класів: {class_names[:8]}")

    image_path = utils.download_demo_image()
    print(f"Демонстраційне зображення завантажено: {image_path}")

    summary = {
        "ultralytics_version": ultralytics.__version__,
        "model_weights": str(model_path.name),
        "num_classes": len(class_names),
        "example_classes": class_names[:10],
        "demo_image": str(image_path),
    }

    out_file = config.DATA_DIR / "step1_setup.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✔ Результати кроку 1 збережено: {out_file}")
    return summary


def main() -> None:
    """Точка входу кроку 1."""
    run_setup()


if __name__ == "__main__":
    main()
