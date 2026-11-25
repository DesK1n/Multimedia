"""Модуль генерації цифрових шумів: адитивного гаусового та імпульсного «сіль та перець»."""

from typing import Optional

import numpy as np


def add_gaussian_noise(
    img: np.ndarray,
    sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""Додає адитивний білий гаусів шум (Additive White Gaussian Noise) до зображення.

    Математична модель:
    $$I_{\text{noisy}}(x, y) = \text{clip}\left(I(x, y) + \mathcal{N}(0, \sigma_{\text{abs}}^2), 0, 255\right)$$
    де $\sigma_{\text{abs}} = \sigma \cdot 255.0$ є абсолютним стандартним відхиленням шуму.

    Args:
        img (np.ndarray): Вхідне цифрове зображення (uint8, 2D або 3D).
        sigma (float): Нормоване стандартне відхилення шуму відносно діапазону [0, 1].
        rng (Optional[np.random.Generator]): Генератор випадкових чисел для відтворюваності.

    Returns:
        np.ndarray: Зашумлене зображення з типом uint8.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    noise = rng.normal(0.0, sigma * 255.0, img.shape)
    noisy_float = img.astype(np.float64) + noise
    return np.clip(np.round(noisy_float), 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    img: np.ndarray,
    amount: float,
    salt_vs_pepper: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""Додає імпульсний біполярний шум «сіль та перець» (Salt & Pepper Noise).

    Частка пікселів $p = \text{amount}$ випадковим чином заміщується на граничні
    значення динамічного діапазону: $255$ («сіль») або $0$ («перець»).

    Args:
        img (np.ndarray): Вхідне цифрове зображення (uint8).
        amount (float): Загальна частка зашумлених пікселів на інтервалі [0, 1].
        salt_vs_pepper (float): Співвідношення білих («сіль») та чорних («перець») імпульсів.
        rng (Optional[np.random.Generator]): Генератор випадкових чисел.

    Returns:
        np.ndarray: Зашумлене зображення з типом uint8.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    noisy = img.copy()
    h, w = img.shape[:2]
    prob_matrix = rng.random((h, w))

    salt_threshold = amount * salt_vs_pepper
    pepper_threshold = amount

    salt_mask = prob_matrix < salt_threshold
    pepper_mask = (prob_matrix >= salt_threshold) & (prob_matrix < pepper_threshold)

    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0

    return noisy
