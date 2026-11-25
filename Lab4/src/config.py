"""Конфігурація лабораторної роботи № 4.

Модуль містить глобальні параметри експериментів, шляхи до каталогів,
гіперпараметри навчання та конфігурацію обчислювального пристрою.
"""

from pathlib import Path
import random
from typing import Final

import numpy as np
import torch

# Базові каталоги
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data_cache"
RESULTS_DIR: Final[Path] = BASE_DIR / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"

# Створення необхідних каталогів
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Відтворюваність експериментів
SEED: Final[int] = 42

# Параметри класів датасету Fashion-MNIST
CLASS_NAMES: Final[list[str]] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
NUM_CLASSES: Final[int] = len(CLASS_NAMES)

# Розмір підвибірки для контрольованого порівняння
TRAIN_PER_CLASS: Final[int] = 600
VAL_PER_CLASS: Final[int] = 150
TEST_PER_CLASS: Final[int] = 150

# Розміри вхідних зображень
CNN_IMAGE_SIZE: Final[int] = 28
PRETRAINED_IMAGE_SIZE: Final[int] = 64
PRETRAINED_CHANNELS: Final[int] = 3

# Гіперпараметри навчання
BATCH_SIZE: Final[int] = 64
EPOCHS: Final[int] = 8
LEARNING_RATE_CNN: Final[float] = 1e-3
LEARNING_RATE_TRANSFER: Final[float] = 1e-3


def set_seed(seed: int = SEED) -> None:
    """Фіксує генератори випадкових чисел для повної відтворюваності.

    Args:
        seed (int): Зерно ініціалізації генераторів.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Визначає доступний обчислювальний пристрій (CUDA, MPS або CPU).

    Returns:
        torch.device: Обчислювальний пристрій для тренування та інференсу.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: Final[torch.device] = get_device()
