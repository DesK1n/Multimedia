"""Модуль покращення контрасту (HE, CLAHE) та бінаризації (Otsu, Adaptive Thresholding)."""

import cv2
import numpy as np


def enhance_contrast_he(gray_img: np.ndarray) -> np.ndarray:
    """Виконує глобальне вирівнювання (еквалізацію) гістограми яскравості.

    Args:
        gray_img (np.ndarray): Одноканальне напівтонове зображення uint8.

    Returns:
        np.ndarray: Еквалізоване зображення.
    """
    if gray_img.ndim != 2:
        raise ValueError("Глобальна еквалізація застосовується до одноканального зображення.")
    return cv2.equalizeHist(gray_img)


def enhance_contrast_clahe(
    gray_img: np.ndarray, clip_limit: float = 2.0, tile_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Застосовує адаптивну гістограмну еквалізацію з обмеженням контрасту (CLAHE).

    Зображення розбивається на локальні тайли tile_size, для кожного з яких
    обчислюється власна гістограма з відсіканням пікових значень clip_limit,
    після чого виконується білінійне згладжування меж тайлів.

    Args:
        gray_img (np.ndarray): Напівтонове зображення uint8.
        clip_limit (float): Границя відсікання локального контрасту.
        tile_size (tuple[int, int]): Розмір локального блоку сітки (M, N).

    Returns:
        np.ndarray: Зображення, оброблене алгоритмом CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def binarize_otsu(gray_img: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Виконує глобальну бінаризацію методом Оцу (Otsu's Thresholding).

    Знаходить глобальний оптимальний поріг $T^*$, який максимізує міжкласову дисперсію:
    $$T^* = \arg\max_T \left\{ \omega_0(T)\omega_1(T) [\mu_0(T) - \mu_1(T)]^2 \right\}$$

    Args:
        gray_img (np.ndarray): Напівтонове зображення uint8.

    Returns:
        tuple[float, np.ndarray]: Оптимальний числовий поріг та бінарне зображення.
    """
    thresh_val, binary = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(thresh_val), binary


def binarize_adaptive(
    gray_img: np.ndarray, block_size: int = 15, c_const: float = 5.0
) -> np.ndarray:
    """Виконує локальну адаптивну бінаризацію за зваженим гаусовим вікном.

    Поріг T(x, y) обчислюється індивідуально для кожного пікселя як зважена
    гаусова сума сусідів у вікні block_size мінус стала константа c_const.

    Args:
        gray_img (np.ndarray): Напівтонове зображення uint8.
        block_size (int): Розмір локального вікна (має бути непарним).
        c_const (float): Константа зсуву від локального середнього.

    Returns:
        np.ndarray: Адаптивно бінаризоване зображення.
    """
    if block_size % 2 == 0:
        raise ValueError(f"block_size має бути непарним числом, отримано {block_size}")
    return cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c_const,
    )


def simulate_uneven_illumination(
    gray_img: np.ndarray, grad_min: float = -80.0, grad_max: float = 80.0
) -> np.ndarray:
    """Симулює нерівномірне градієнтне освітлення сцени (зліва затемнення, справа засвічення).

    Args:
        gray_img (np.ndarray): Початкове напівтонове зображення.
        grad_min (float): Мінімальне значення адитивного градієнта (лівий край).
        grad_max (float): Максимальне значення адитивного градієнта (правий край).

    Returns:
        np.ndarray: Зображення з нерівномірним штучним освітленням.
    """
    h, w = gray_img.shape[:2]
    gradient_row = np.linspace(grad_min, grad_max, w)
    gradient_2d = np.tile(gradient_row, (h, 1))
    uneven = np.clip(gray_img.astype(np.float64) + gradient_2d, 0, 255).astype(np.uint8)
    return uneven
