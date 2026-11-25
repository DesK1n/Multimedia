"""Модуль підготовки та завантаження даних Fashion-MNIST.

Забезпечує завантаження, збалансовану вибірку (Stratified Subset) та
конструювання пайплайнів попередньої обробки й аугментації даних.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import config


@dataclass
class Datasets:
    """Контейнер із готовими завантажувачами даних для експерименту.

    Attributes:
        train_loader: DataLoader для навчальної підвибірки.
        val_loader: DataLoader для валідаційної підвибірки.
        test_loader: DataLoader для тестової підвибірки.
        train_size: Кількість зразків у навчальній підвибірці.
        val_size: Кількість зразків у валідаційній підвибірці.
        test_size: Кількість зразків у тестовій підвибірці.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    val_size: int
    test_size: int


def _to_3_channels(x: torch.Tensor) -> torch.Tensor:
    """Дублює одноканальний тензор у три канали (C=3)."""
    return x.repeat(3, 1, 1)


def load_raw_fashion_mnist() -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Завантажує оригінальні підвибірки Fashion-MNIST без трансформацій.

    Returns:
        Tuple[datasets.FashionMNIST, datasets.FashionMNIST]: Кортеж із
        навчального та тестового датасетів.
    """
    train_raw = datasets.FashionMNIST(
        root=str(config.DATA_DIR), train=True, download=True
    )
    test_raw = datasets.FashionMNIST(
        root=str(config.DATA_DIR), train=False, download=True
    )
    return train_raw, test_raw


def get_balanced_indices(
    targets: torch.Tensor, per_class: int, seed: int
) -> np.ndarray:
    """Генерує збалансовані індекси зразків для кожного класу.

    Args:
        targets: Тензор міток класів усього датасету.
        per_class: Кількість зразків для відбору з кожного класу.
        seed: Базове зерно генератора псевдовипадкових чисел.

    Returns:
        np.ndarray: Масив перемішаних індексів збалансованої підвибірки.
    """
    rng = np.random.default_rng(seed)
    targets_np = targets.numpy()
    selected_indices = []

    for class_idx in range(config.NUM_CLASSES):
        class_indices = np.where(targets_np == class_idx)[0]
        rng.shuffle(class_indices)
        selected_indices.append(class_indices[:per_class])

    result = np.concatenate(selected_indices)
    rng.shuffle(result)
    return result


def get_balanced_train_val_indices(
    targets: torch.Tensor,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Розбиває навчальний набір на неперетинні збалансовані train та val індекси.

    Args:
        targets: Тензор міток класів навчального датасету.
        train_per_class: Кількість навчальних зразків на клас.
        val_per_class: Кількість валідаційних зразків на клас.
        seed: Зерно генератора псевдовипадкових чисел.

    Returns:
        tuple[np.ndarray, np.ndarray]: Масиви train та val індексів.
    """
    rng = np.random.default_rng(seed)
    targets_np = targets.numpy()
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_idx in range(config.NUM_CLASSES):
        class_indices = np.where(targets_np == class_idx)[0]
        rng.shuffle(class_indices)
        train_parts.append(class_indices[:train_per_class])
        val_parts.append(
            class_indices[train_per_class : train_per_class + val_per_class]
        )

    train_res = np.concatenate(train_parts)
    val_res = np.concatenate(val_parts)
    rng.shuffle(train_res)
    rng.shuffle(val_res)
    return train_res, val_res


def build_transform(
    image_size: int,
    channels: int,
    augment: bool,
    aug_mode: str = "combined",
) -> transforms.Compose:
    """Формує пайплайн попередньої обробки та аугментації зображень.

    Args:
        image_size: Цільовий просторовий розмір зображення (квадрат).
        channels: Кількість каналів (1 для сірого, 3 для псевдо-RGB).
        augment: Чи застосовувати випадкові перетворення (Data Augmentation).
        aug_mode: Режим аугментації ('none', 'flip', 'rotation', 'affine', 'combined').

    Returns:
        transforms.Compose: Скомпонована послідовність трансформацій torchvision.
    """
    transform_steps: list = [transforms.Resize((image_size, image_size))]

    effective_mode = aug_mode if augment else "none"
    if effective_mode in ("flip", "combined"):
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if effective_mode in ("rotation", "combined"):
        transform_steps.append(transforms.RandomRotation(degrees=10))
    if effective_mode == "affine":
        transform_steps.append(
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08))
        )

    transform_steps.append(transforms.ToTensor())

    if channels == 3:
        # Дублюємо 1-канальне зображення до 3 каналів для моделей ImageNet
        transform_steps.append(transforms.Lambda(_to_3_channels))
        transform_steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )
    else:
        # Нормалізація для одноканального зображення
        transform_steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_steps)


def build_datasets(
    *,
    image_size: int,
    channels: int,
    augment: bool,
    aug_mode: str = "combined",
) -> Datasets:
    """Створює збалансовані DataLoader'и (train/val/test) для заданого експерименту.

    Args:
        image_size: Розмірність вхідного зображення.
        channels: Кількість каналів (1 або 3).
        augment: Прапорець активації аугментації навчальної вибірки.
        aug_mode: Тип аугментації ('none', 'flip', 'rotation', 'affine', 'combined').

    Returns:
        Datasets: Об'єкт контейнера зі сформованими DataLoader'ами.
    """
    train_raw, test_raw = load_raw_fashion_mnist()

    train_transform = build_transform(
        image_size, channels, augment=augment, aug_mode=aug_mode
    )
    eval_transform = build_transform(image_size, channels, augment=False)

    train_full = datasets.FashionMNIST(
        root=str(config.DATA_DIR),
        train=True,
        download=False,
        transform=train_transform,
    )
    val_full = datasets.FashionMNIST(
        root=str(config.DATA_DIR),
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_full = datasets.FashionMNIST(
        root=str(config.DATA_DIR),
        train=False,
        download=False,
        transform=eval_transform,
    )

    train_indices, val_indices = get_balanced_train_val_indices(
        train_raw.targets,
        train_per_class=config.TRAIN_PER_CLASS,
        val_per_class=config.VAL_PER_CLASS,
        seed=config.SEED,
    )
    test_indices = get_balanced_indices(
        test_raw.targets, config.TEST_PER_CLASS, seed=config.SEED + 1
    )

    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(val_full, val_indices)
    test_subset = Subset(test_full, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return Datasets(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=len(train_subset),
        val_size=len(val_subset),
        test_size=len(test_subset),
    )
