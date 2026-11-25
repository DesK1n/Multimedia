"""Конфігураційні параметри та константи для Лабораторних робіт №1 та №2."""

from pathlib import Path

# Базові каталоги
BASE_DIR: Path = Path(__file__).resolve().parent.parent
IMAGES_DIR: Path = BASE_DIR / "images"
RESULTS_DIR: Path = BASE_DIR / "results" / "lab1"
RESULTS_DIR_LAB2: Path = BASE_DIR / "results" / "lab2"

# ЛР №1: Рівні якості для експерименту з JPEG-компресією
QUALITY_LEVELS: list[int] = [10, 20, 40, 60, 80, 95]
RESOLUTION_SCALES: list[int] = [1, 2, 4]
FIXED_RESOLUTION_QUALITY: int = 40
ROTATION_ANGLE_DEG: float = 30.0
CROP_RATIO: float = 0.6

# ЛР №2: Рівні шумів та параметри фільтрації
NOISE_SIGMAS: list[float] = [0.05, 0.1, 0.2]
GAUSSIAN_KERNEL_SIZES: list[int] = [3, 5, 7, 9, 11]
MEDIAN_KERNEL_SIZES: list[int] = [3, 5, 7, 9, 11]
BILATERAL_DIAMETERS: list[int] = [5, 9, 13, 17]


def ensure_directories() -> None:
    """Створює необхідні каталоги для вхідних даних та результатів, якщо вони відсутні."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_LAB2.mkdir(parents=True, exist_ok=True)
