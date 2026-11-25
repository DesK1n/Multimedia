"""Модуль просторових геометричних перетворень та дослідження інтерполяцій."""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.metrics import psnr_manual, mse_manual


def perform_geometry_transforms(
    img_bgr: np.ndarray,
    angle_deg: float = 30.0,
    crop_ratio: float = 0.6,
    scale_down: float = 0.5,
    scale_up: float = 1.5,
) -> dict[str, np.ndarray]:
    """Виконує поворот, кадрування та масштабування зображення.

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення.
        angle_deg (float): Кут обертання навколо центра у градусах.
        crop_ratio (float): Частка розмірів для центрального обрізання.
        scale_down (float): Коефіцієнт зменшення.
        scale_up (float): Коефіцієнт збільшення.

    Returns:
        dict[str, np.ndarray]: Словник трансформованих зображень.
    """
    h, w = img_bgr.shape[:2]

    # 1. Поворот на кут навколо геометричного центра
    center = (w / 2.0, h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, angle=angle_deg, scale=1.0)
    rotated = cv2.warpAffine(
        img_bgr, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    # 2. Обрізання (кадрування) центральної області
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = img_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]

    # 3. Масштабування
    scaled_down = cv2.resize(
        img_bgr,
        (int(w * scale_down), int(h * scale_down)),
        interpolation=cv2.INTER_LINEAR,
    )
    scaled_up = cv2.resize(
        img_bgr,
        (int(w * scale_up), int(h * scale_up)),
        interpolation=cv2.INTER_LINEAR,
    )

    return {
        "original": img_bgr,
        "rotated": rotated,
        "cropped": cropped,
        "scaled_down": scaled_down,
        "scaled_up": scaled_up,
    }


def compare_interpolations(
    img_bgr: np.ndarray, output_dir: Path, downscale_factor: int = 4
) -> dict[str, dict[str, float | np.ndarray]]:
    r"""Досліджує деградацію якості при циклі зменшення/збільшення для різних інтерполяцій.

    Порівнюються методи:
    - INTER_NEAREST (найближчий сусід)
    - INTER_LINEAR (білінійна інтерполяція)
    - INTER_CUBIC (бікубічна інтерполяція $4 \times 4$)
    - INTER_LANCZOS4 (інтерполяція Ланцоша у вікні $8 \times 8$)

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення.
        output_dir (Path): Каталог для збереження результатів.
        downscale_factor (int): Фактор тимчасового зменшення розміру.

    Returns:
        dict[str, dict[str, float | np.ndarray]]: Результати за кожним методом.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_bgr.shape[:2]
    small_size = (w // downscale_factor, h // downscale_factor)

    methods = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }

    results: dict[str, dict[str, float | np.ndarray]] = {}
    csv_rows = []

    for name, flag in methods.items():
        small = cv2.resize(img_bgr, small_size, interpolation=flag)
        restored = cv2.resize(small, (w, h), interpolation=flag)

        mse_val = mse_manual(img_bgr, restored)
        psnr_val = psnr_manual(img_bgr, restored)

        results[name] = {
            "mse": mse_val,
            "psnr": psnr_val,
            "restored_img": restored,
        }
        csv_rows.append([name, f"{mse_val:.2f}", f"{psnr_val:.2f}"])

    csv_path = output_dir / "03_interpolation_psnr.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Метод інтерполяції", "MSE після циклу", "PSNR після циклу, дБ"])
        writer.writerows(csv_rows)

    print(f"[INFO] Результати порівняння інтерполяцій збережено у: {csv_path}")
    return results
