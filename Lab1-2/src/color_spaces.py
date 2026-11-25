"""Модуль дослідження колірних просторів: RGB, Grayscale, HSV, YCbCr."""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.metrics import mse_manual


def convert_to_grayscale_manual(img_bgr: np.ndarray) -> np.ndarray:
    r"""Перетворює 3-канальне BGR-зображення у напівтонове за фізіологічною формулою.

    Формула:
    $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення (uint8).

    Returns:
        np.ndarray: Одноканальне Grayscale зображення (uint8).
    """
    b, g, r = cv2.split(img_bgr)
    gray_float = (
        0.299 * r.astype(np.float64)
        + 0.587 * g.astype(np.float64)
        + 0.114 * b.astype(np.float64)
    )
    return np.clip(np.round(gray_float), 0, 255).astype(np.uint8)


def analyze_color_spaces(
    img_bgr: np.ndarray, output_dir: Path
) -> dict[str, float | np.ndarray]:
    """Досліджує декомпозицію на канали, точність Grayscale та оборотність YCbCr.

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення.
        output_dir (Path): Каталог для збереження результатів.

    Returns:
        dict[str, float | np.ndarray]: Словник числових метрик та матриць каналів.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Розбиття на канали BGR
    b, g, r = cv2.split(img_bgr)

    # 2. Перетворення у Grayscale
    gray_manual = convert_to_grayscale_manual(img_bgr)
    gray_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    diff_matrix = np.abs(
        gray_manual.astype(np.int16) - gray_cv2.astype(np.int16)
    ).astype(np.uint8)

    mean_diff = float(np.mean(diff_matrix))
    max_diff = int(np.max(diff_matrix))

    # 3. Перетворення у HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 4. Перетворення у YCbCr (у OpenCV це COLOR_BGR2YCrCb)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

    # 5. Оборотність RGB -> YCrCb -> RGB
    bgr_reconstructed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    mse_roundtrip = mse_manual(img_bgr, bgr_reconstructed)

    # Збереження числових результатів у CSV
    csv_path = output_dir / "02_color_spaces.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Показник", "Значення", "Фізичне/математичне тлумачення"])
        writer.writerow([
            "Середня абсолютна різниця Grayscale",
            f"{mean_diff:.4f}",
            "Похибка округлення дійсних чисел у цілі uint8",
        ])
        writer.writerow([
            "Максимальна абсолютна різниця Grayscale",
            f"{max_diff}",
            "Максимальне відхилення між ручним та апаратним округленням",
        ])
        writer.writerow([
            "MSE циклу RGB -> YCbCr -> RGB",
            f"{mse_roundtrip:.4f}",
            "Підтвердження практичної оборотності колірного перетворення",
        ])

    print(f"[INFO] Результати аналізу колірних просторів збережено у: {csv_path}")

    return {
        "mean_diff_gray": mean_diff,
        "max_diff_gray": max_diff,
        "mse_roundtrip": mse_roundtrip,
        "r": r,
        "g": g,
        "b": b,
        "gray_manual": gray_manual,
        "gray_cv2": gray_cv2,
        "diff_matrix": diff_matrix,
        "h": h_ch,
        "s": s_ch,
        "v": v_ch,
        "y": y_ch,
        "cr": cr_ch,
        "cb": cb_ch,
    }
