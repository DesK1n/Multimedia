"""Конфігурація лабораторної роботи № 5.

Централізоване визначення шляхів до каталогів, файлів моделей YOLOv8,
тестових мультимедійних зразків, параметрів досліджень та порогів відсікання.
"""

from pathlib import Path
from typing import Final, List

# Базові каталоги проєкту
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
SRC_DIR: Final[Path] = BASE_DIR / "src"
WEIGHTS_DIR: Final[Path] = BASE_DIR / "weights"
RESULTS_DIR: Final[Path] = BASE_DIR / "results"
IMAGES_DIR: Final[Path] = RESULTS_DIR / "images"
PLOTS_DIR: Final[Path] = RESULTS_DIR / "plots"
VIDEO_DIR: Final[Path] = RESULTS_DIR / "video"
DATA_DIR: Final[Path] = RESULTS_DIR / "data"

for directory in (
    WEIGHTS_DIR,
    RESULTS_DIR,
    IMAGES_DIR,
    PLOTS_DIR,
    VIDEO_DIR,
    DATA_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Шляхи до ваг попередньо навчених моделей YOLOv8
YOLO_NANO_WEIGHTS: Final[Path] = WEIGHTS_DIR / "yolov8n.pt"
YOLO_SMALL_WEIGHTS: Final[Path] = WEIGHTS_DIR / "yolov8s.pt"

# Тестові зображення Ultralytics
DEMO_IMAGE_URL: Final[str] = "https://ultralytics.com/images/bus.jpg"
DEMO_IMAGE_PATH: Final[Path] = IMAGES_DIR / "bus.jpg"

VIDEO_SOURCE_URL: Final[str] = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"
    "vtest.avi"
)
DEMO_VIDEO_URL: Final[str] = VIDEO_SOURCE_URL
DEMO_VIDEO_PATH: Final[Path] = VIDEO_DIR / "vtest.avi"
OUTPUT_VIDEO_PATH: Final[Path] = VIDEO_DIR / "video_detection.mp4"

# Набір даних для розрахунку mAP (COCO128)
VAL_DATASET: Final[str] = "coco128.yaml"

# Дослідницькі пороги впевненості (Confidence Thresholds)
CONFIDENCE_THRESHOLDS: Final[List[float]] = [0.1, 0.25, 0.5, 0.75, 0.9]

# Дослідницькі роздільні здатності для аналізу швидкодії (FPS)
IMAGE_SIZES: Final[List[int]] = [320, 480, 640, 960, 1280]

# Відтворюваність
RANDOM_SEED: Final[int] = 42
