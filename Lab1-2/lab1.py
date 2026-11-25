"""Головний модуль запуску повного конвеєра Лабораторної роботи №1.

Дисципліна: "Обробка мультимедійних даних"
Тема: "Первинна обробка та представлення мультимедійних даних"

Скрипт послідовно виконує всі етапи:
1. Завантаження зображення та збереження у форматах BMP, PNG, JPG (з розрахунком розмірів).
2. Дослідження колірних просторів: RGB, Grayscale (ручна формула vs cv2), HSV, YCbCr.
3. Геометричні перетворення: поворот, кадрування, масштабування та аналіз інтерполяцій.
4. Верифікація власних метрик якості MSE, RMSE, PSNR, SSIM проти scikit-image.
5. Дослідницький експеримент із JPEG-компресією Q in {10, 20, 40, 60, 80, 95}.
6. Дослідження впливу зміни просторової роздільної здатності на якість при Q=40.
"""

import csv
import sys
from pathlib import Path

# Додаємо поточну директорію у sys.path для підтримки імпорту src
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

from src.color_spaces import analyze_color_spaces
from src.config import (
    FIXED_RESOLUTION_QUALITY,
    QUALITY_LEVELS,
    RESOLUTION_SCALES,
    RESULTS_DIR,
    ROTATION_ANGLE_DEG,
    CROP_RATIO,
    ensure_directories,
)
from src.experiments import (
    run_jpeg_quality_experiment,
    run_resolution_experiment,
)
from src.geometry import (
    compare_interpolations,
    perform_geometry_transforms,
)
from src.io_formats import (
    load_source_image,
    robust_imread,
    save_formats_and_compare,
)
from src.metrics import compare_manual_vs_skimage
from src.plotting import (
    plot_color_channels,
    plot_geometry_transforms,
    plot_grayscale_comparison,
    plot_interpolation_comparison,
    plot_jpeg_artifacts,
    plot_jpeg_experiment,
    plot_metrics_validation,
)


def run_pipeline() -> None:
    """Виконує наскрізний експериментальний пайплайн Лабораторної роботи №1."""
    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №1: ПЕРВИННА ОБРОБКА МУЛЬТИМЕДІЙНИХ ДАНИХ")
    print("=" * 70)

    ensure_directories()

    # 0. Завантаження вхідного зображення
    img_bgr = load_source_image()
    h, w, c = img_bgr.shape
    print(f"\n[Крок 0] Параметри зображення: висота={h}, ширина={w}, канали={c}")

    # 1. Формати збереження даних
    print("\n[Крок 1] Дослідження форматів файлів (BMP, PNG, JPG)...")
    sizes_info = save_formats_and_compare(img_bgr, RESULTS_DIR)
    for name, sz in sizes_info.items():
        print(f"  * {name}: {sz:.2f} КБ")

    # 2. Колірні простори
    print("\n[Крок 2] Дослідження колірних просторів (RGB, Grayscale, HSV, YCbCr)...")
    color_results = analyze_color_spaces(img_bgr, RESULTS_DIR)
    print(f"  * Середня різниця Grayscale (ручна vs cv2): {color_results['mean_diff_gray']:.4f}")
    print(f"  * MSE циклу RGB -> YCbCr -> RGB: {color_results['mse_roundtrip']:.4f}")

    # Побудова графіків для колірних просторів
    plot_color_channels(
        {
            "r": color_results["r"],
            "g": color_results["g"],
            "b": color_results["b"],
            "h": color_results["h"],
            "s": color_results["s"],
            "v": color_results["v"],
            "y": color_results["y"],
            "cr": color_results["cr"],
            "cb": color_results["cb"],
        },
        RESULTS_DIR / "02_color_channels.png",
    )
    plot_grayscale_comparison(
        color_results["gray_manual"],
        color_results["gray_cv2"],
        color_results["diff_matrix"],
        RESULTS_DIR / "02_grayscale_compare.png",
    )

    # 3. Геометричні трансформації та інтерполяція
    print("\n[Крок 3] Геометричні перетворення та порівняння методів інтерполяції...")
    geom_transforms = perform_geometry_transforms(
        img_bgr, angle_deg=ROTATION_ANGLE_DEG, crop_ratio=CROP_RATIO
    )
    plot_geometry_transforms(geom_transforms, RESULTS_DIR / "03_geometry.png")

    interp_results = compare_interpolations(img_bgr, RESULTS_DIR, downscale_factor=4)
    plot_interpolation_comparison(
        interp_results, RESULTS_DIR / "03_interpolation_compare.png"
    )
    for name, metrics in interp_results.items():
        print(f"  * {name}: PSNR={metrics['psnr']:.2f} дБ, MSE={metrics['mse']:.2f}")

    # 4. Метрики якості та їх верифікація
    print("\n[Крок 4] Валідація власних метрик (MSE, RMSE, PSNR, SSIM) проти scikit-image...")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 15.0, gray.shape)
    noisy_gray = np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    metrics_comp = compare_manual_vs_skimage(gray, noisy_gray)
    metrics_csv_path = RESULTS_DIR / "04_metrics_validation.csv"
    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика", "Власна реалізація (NumPy)", "scikit-image", "Абсолютна похибка"])
        for m_name, m_vals in metrics_comp.items():
            print(
                f"  * {m_name:4s}: Власна={m_vals['manual']:8.4f} | "
                f"skimage={m_vals['library']:8.4f} | |Diff|={m_vals['abs_diff']:.2e}"
            )
            writer.writerow([
                m_name,
                f"{m_vals['manual']:.4f}",
                f"{m_vals['library']:.4f}",
                f"{m_vals['abs_diff']:.2e}",
            ])

    # Отримання повної просторової карти SSIM для візуалізації
    _, ssim_map = sk_ssim(gray, noisy_gray, data_range=255, full=True)
    plot_metrics_validation(
        noisy_gray, ssim_map, RESULTS_DIR / "04_metrics_validation.png"
    )

    # 5. Дослідницький експеримент: JPEG-компресія
    print("\n[Крок 5] Дослідницький експеримент із JPEG-компресією Q in {10..95}...")
    jpeg_results = run_jpeg_quality_experiment(img_bgr, QUALITY_LEVELS, RESULTS_DIR)
    plot_jpeg_experiment(jpeg_results, RESULTS_DIR / "05_jpeg_psnr_ssim.png")

    low_q_path = RESULTS_DIR / f"05_jpeg_q{QUALITY_LEVELS[0]}.jpg"
    low_q_img = robust_imread(low_q_path, cv2.IMREAD_COLOR)
    if low_q_img is not None:
        plot_jpeg_artifacts(img_bgr, low_q_img, RESULTS_DIR / "05_jpeg_artifacts.png")

    for row in jpeg_results:
        print(
            f"  * Q={int(row['Q']):2d} -> Розмір={row['size_kb']:6.2f} КБ | "
            f"PSNR={row['psnr']:5.2f} дБ | SSIM={row['ssim']:.4f}"
        )

    # 6. Вплив просторової роздільної здатності при фіксованому Q
    print(
        f"\n[Крок 6] Вплив зміни роздільної здатності при фіксованій якості Q={FIXED_RESOLUTION_QUALITY}..."
    )
    res_results = run_resolution_experiment(
        img_bgr, RESOLUTION_SCALES, FIXED_RESOLUTION_QUALITY, RESULTS_DIR
    )
    for row in res_results:
        print(
            f"  * Масштаб 1/{int(row['scale'])}x ({int(row['width'])}x{int(row['height'])}) "
            f"-> Розмір={row['size_kb']:6.2f} КБ | PSNR={row['psnr']:.2f} дБ"
        )

    print("\n" + "=" * 70)
    print(f"[УСПІХ] Лабораторна робота №1 виконана! Усі файли збережено у: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
