"""Модуль проведення дослідницьких експериментів з JPEG-компресією."""

import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

from src.io_formats import robust_imread, robust_imwrite
from src.metrics import mse_manual, rmse_manual, psnr_manual


def run_jpeg_quality_experiment(
    img_bgr: np.ndarray, quality_levels: list[int], output_dir: Path
) -> list[dict[str, float]]:
    """Досліджує залежність якості відновлення та розміру файлу від коефіцієнта Q.

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення.
        quality_levels (list[int]): Список коефіцієнтів якості (наприклад, [10, 20, 40, 60, 80, 95]).
        output_dir (Path): Каталог для збереження стиснених файлів та CSV.

    Returns:
        list[dict[str, float]]: Таблиця результатів для кожного рівня якості.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for q in quality_levels:
        jpg_path = output_dir / f"05_jpeg_q{q}.jpg"
        robust_imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        file_size_kb = jpg_path.stat().st_size / 1024.0

        decoded = robust_imread(jpg_path, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"Не вдалося зчитати збережений файл: {jpg_path}")

        mse_val = mse_manual(img_bgr, decoded)
        rmse_val = rmse_manual(img_bgr, decoded)
        psnr_val = psnr_manual(img_bgr, decoded)
        ssim_val = float(sk_ssim(img_bgr, decoded, channel_axis=2, data_range=255))

        rows.append({
            "Q": float(q),
            "size_kb": file_size_kb,
            "mse": mse_val,
            "rmse": rmse_val,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    csv_path = output_dir / "05_jpeg_experiment.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q (якість)", "Розмір, КБ", "MSE", "RMSE", "PSNR, дБ", "SSIM"])
        for r in rows:
            writer.writerow([
                int(r["Q"]),
                f"{r['size_kb']:.2f}",
                f"{r['mse']:.2f}",
                f"{r['rmse']:.2f}",
                f"{r['psnr']:.2f}",
                f"{r['ssim']:.4f}",
            ])

    print(f"[INFO] Результати експерименту JPEG збережено у: {csv_path}")
    return rows


def run_resolution_experiment(
    img_bgr: np.ndarray, scales: list[int], quality: int, output_dir: Path
) -> list[dict[str, float]]:
    """Досліджує вплив зміни роздільної здатності на якість компресії при фіксованому Q.

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення.
        scales (list[int]): Список коефіцієнтів зменшення (наприклад, [1, 2, 4]).
        quality (int): Фіксований рівень якості (наприклад, 40).
        output_dir (Path): Каталог для збереження результатів.

    Returns:
        list[dict[str, float]]: Список результатів експерименту.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_bgr.shape[:2]
    rows: list[dict[str, float]] = []

    for s in scales:
        target_size = (w // s, h // s)
        resized_orig = cv2.resize(
            img_bgr, target_size, interpolation=cv2.INTER_AREA
        )

        jpg_path = output_dir / f"05_res_scale_{s}x_q{quality}.jpg"
        robust_imwrite(jpg_path, resized_orig, [cv2.IMWRITE_JPEG_QUALITY, quality])

        decoded = robust_imread(jpg_path, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"Не вдалося зчитати збережений файл: {jpg_path}")

        restored = (
            decoded
            if s == 1
            else cv2.resize(decoded, (w, h), interpolation=cv2.INTER_CUBIC)
        )
        psnr_val = psnr_manual(img_bgr, restored)
        ssim_val = float(
            sk_ssim(img_bgr, restored, channel_axis=2, data_range=255)
        )
        size_kb = jpg_path.stat().st_size / 1024.0

        rows.append({
            "scale": float(s),
            "width": float(target_size[0]),
            "height": float(target_size[1]),
            "size_kb": size_kb,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    csv_path = output_dir / "05_resolution_vs_quality.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Масштаб",
            "Роздільність (WxH)",
            "Розмір, КБ",
            "PSNR (відн. 512x512, Q=40), дБ",
            "SSIM",
        ])
        for r in rows:
            writer.writerow([
                f"1/{int(r['scale'])}x",
                f"{int(r['width'])}x{int(r['height'])}",
                f"{r['size_kb']:.2f}",
                f"{r['psnr']:.2f}",
                f"{r['ssim']:.4f}",
            ])

    print(f"[INFO] Результати впливу роздільної здатності збережено у: {csv_path}")
    return rows
