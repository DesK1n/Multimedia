"""Головний модуль запуску повного конвеєра Лабораторної роботи №2.

Дисципліна: "Обробка мультимедійних даних"
Тема: "Фільтрація, покращення та сегментація зображень"

Скрипт послідовно виконує всі етапи:
1. Генерація шуму: адитивний гаусів та імпульсний «сіль та перець».
2. Просторова фільтрація: Gaussian, Median, Bilateral фільтри.
3. Покращення контрасту: Histogram Equalization (HE) проти CLAHE та побудова гістограм.
4. Сегментація/бінаризація: метод Оцу проти Adaptive Thresholding (при рівномірному та градієнтному освітленні).
5. Дослідницький експеримент: порівняння метрик MSE, PSNR, SSIM для sigma in {0.05, 0.1, 0.2} на двох типах шумів.
6. Міні-дослідження: автоматичний підбір оптимальних параметрів фільтрів за критерієм максимуму PSNR.
"""

import csv
import sys
from pathlib import Path

# Додаємо поточну директорію у sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

from src.config import (
    BILATERAL_DIAMETERS,
    GAUSSIAN_KERNEL_SIZES,
    MEDIAN_KERNEL_SIZES,
    NOISE_SIGMAS,
    RESULTS_DIR_LAB2,
    ensure_directories,
)
from src.enhancement import (
    binarize_adaptive,
    binarize_otsu,
    enhance_contrast_clahe,
    enhance_contrast_he,
    simulate_uneven_illumination,
)
from src.filters import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_median_filter,
    auto_tune_filter_params,
)
from src.io_formats import load_source_image
from src.metrics import mse_manual, psnr_manual
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise
from src.plotting_lab2 import (
    plot_auto_tune,
    plot_contrast_comparison,
    plot_filters_comparison,
    plot_noise_examples,
    plot_psnr_vs_sigma,
    plot_thresholding_comparison,
)


def compute_ssim_robust(img1: np.ndarray, img2: np.ndarray) -> float:
    """Обчислює SSIM для 2D або 3D зображень."""
    if img1.ndim == 3:
        return float(sk_ssim(img1, img2, data_range=255, channel_axis=2))
    return float(sk_ssim(img1, img2, data_range=255))


def run_pipeline_lab2() -> None:
    """Виконує наскрізний експериментальний пайплайн Лабораторної роботи №2."""
    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №2: ФІЛЬТРАЦІЯ, ПОКРАЩЕННЯ ТА СЕГМЕНТАЦІЯ ЗОБРАЖЕНЬ")
    print("=" * 70)

    ensure_directories()

    # 0. Завантаження вихідного зображення
    img_bgr = load_source_image()
    h, w, c = img_bgr.shape
    print(f"\n[Крок 0] Завантажено зображення: {w}x{h}, канали={c}")

    # 1. Генерація шуму
    print("\n[Крок 1] Генерація гаусового та імпульсного шуму...")
    rng = np.random.default_rng(seed=42)
    gaussian_demo = add_gaussian_noise(img_bgr, sigma=0.10, rng=rng)
    sp_demo = add_salt_and_pepper_noise(img_bgr, amount=0.05, rng=rng)

    plot_noise_examples(
        img_bgr,
        gaussian_demo,
        sp_demo,
        RESULTS_DIR_LAB2 / "01_noise_examples.png",
    )

    # 2. Фільтрація: Gaussian, Median, Bilateral
    print("\n[Крок 2] Дослідження лінійних та рангових фільтрів...")
    g_filtered = apply_gaussian_filter(gaussian_demo, ksize=5, sigma=0.0)
    b_filtered = apply_bilateral_filter(gaussian_demo, d=9, sigma_color=75.0, sigma_space=75.0)
    m_filtered_sp = apply_median_filter(sp_demo, ksize=5)
    m_filtered_g = apply_median_filter(gaussian_demo, ksize=5)

    filter_metrics = [
        ("Шум без фільтра (гаусів)", psnr_manual(img_bgr, gaussian_demo)),
        ("Gaussian filter (на гаусів)", psnr_manual(img_bgr, g_filtered)),
        ("Bilateral filter (на гаусів)", psnr_manual(img_bgr, b_filtered)),
        ("Шум без фільтра (сіль-перець)", psnr_manual(img_bgr, sp_demo)),
        ("Median filter (на сіль-перець)", psnr_manual(img_bgr, m_filtered_sp)),
        ("Median filter (на гаусів шум)", psnr_manual(img_bgr, m_filtered_g)),
    ]

    for name, p_val in filter_metrics:
        print(f"  * {name:30s} -> PSNR = {p_val:.2f} дБ")

    filters_csv_path = RESULTS_DIR_LAB2 / "02_filters_psnr.csv"
    with open(filters_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Варіант фільтрації", "PSNR, дБ"])
        for name, p_val in filter_metrics:
            writer.writerow([name, f"{p_val:.2f}"])

    plot_filters_comparison(
        [
            (gaussian_demo, "Гаусів шум (вхідний)"),
            (g_filtered, "Gaussian filter (5x5)"),
            (b_filtered, "Bilateral filter (d=9)"),
            (sp_demo, "Сіль-перець (вхідний)"),
            (m_filtered_sp, "Median filter (на сіль-перець)"),
            (m_filtered_g, "Median filter (на гаусів)"),
        ],
        RESULTS_DIR_LAB2 / "02_filters_compare.png",
    )

    # 3. Покращення контрасту: HE vs CLAHE
    print("\n[Крок 3] Покращення контрасту: Histogram Equalization vs CLAHE...")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    he_img = enhance_contrast_he(gray)
    clahe_img = enhance_contrast_clahe(gray, clip_limit=2.0, tile_size=(8, 8))

    plot_contrast_comparison(
        gray, he_img, clahe_img, RESULTS_DIR_LAB2 / "03_contrast_compare.png"
    )

    # 4. Бінаризація: Otsu vs Adaptive Thresholding
    print("\n[Крок 4] Дослідження бінаризації: Otsu vs Adaptive Thresholding...")
    t_unif, otsu_unif = binarize_otsu(gray)
    adapt_unif = binarize_adaptive(gray, block_size=15, c_const=5.0)

    uneven_gray = simulate_uneven_illumination(gray, grad_min=-80.0, grad_max=80.0)
    t_uneven, otsu_uneven = binarize_otsu(uneven_gray)
    adapt_uneven = binarize_adaptive(uneven_gray, block_size=15, c_const=5.0)
    print(f"  * Поріг Оцу (рівномірне освітлення): T* = {t_unif:.1f}")
    print(f"  * Поріг Оцу (нерівномірний градієнт): T* = {t_uneven:.1f}")

    plot_thresholding_comparison(
        gray,
        otsu_unif,
        adapt_unif,
        uneven_gray,
        otsu_uneven,
        adapt_uneven,
        RESULTS_DIR_LAB2 / "04_thresholding_compare.png",
        otsu_t_unif=t_unif,
        otsu_t_uneven=t_uneven,
    )

    # 5. Дослідницький експеримент: PSNR = f(sigma), SSIM = f(sigma)
    print("\n[Крок 5] Дослідницький експеримент для sigma in {0.05, 0.10, 0.20}...")
    research_rows: list[dict] = []
    exp_rng = np.random.default_rng(seed=123)

    for sigma in NOISE_SIGMAS:
        # Гаусів шум
        noisy_g = add_gaussian_noise(img_bgr, sigma=sigma, rng=exp_rng)
        filt_g = apply_gaussian_filter(noisy_g, ksize=5, sigma=0.0)
        filt_m_g = apply_median_filter(noisy_g, ksize=5)
        filt_b_g = apply_bilateral_filter(noisy_g, d=9, sigma_color=75.0, sigma_space=75.0)

        for f_name, cur_img in [
            ("без фільтра", noisy_g),
            ("Gaussian", filt_g),
            ("Median", filt_m_g),
            ("Bilateral", filt_b_g),
        ]:
            research_rows.append({
                "noise_type": "гаусів",
                "sigma": sigma,
                "filter": f_name,
                "mse": mse_manual(img_bgr, cur_img),
                "psnr": psnr_manual(img_bgr, cur_img),
                "ssim": compute_ssim_robust(img_bgr, cur_img),
            })

        # Імпульсний шум «сіль та перець» (amount = sigma для зіставності інтенсивності)
        noisy_sp = add_salt_and_pepper_noise(img_bgr, amount=sigma, rng=exp_rng)
        filt_g_sp = apply_gaussian_filter(noisy_sp, ksize=5, sigma=0.0)
        filt_m_sp = apply_median_filter(noisy_sp, ksize=5)
        filt_b_sp = apply_bilateral_filter(noisy_sp, d=9, sigma_color=75.0, sigma_space=75.0)

        for f_name, cur_img in [
            ("без фільтра", noisy_sp),
            ("Gaussian", filt_g_sp),
            ("Median", filt_m_sp),
            ("Bilateral", filt_b_sp),
        ]:
            research_rows.append({
                "noise_type": "сіль-перець",
                "sigma": sigma,
                "filter": f_name,
                "mse": mse_manual(img_bgr, cur_img),
                "psnr": psnr_manual(img_bgr, cur_img),
                "ssim": compute_ssim_robust(img_bgr, cur_img),
            })

    # Збереження таблиці
    exp_csv_path = RESULTS_DIR_LAB2 / "05_research_experiment.csv"
    with open(exp_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Тип шуму", "sigma", "Фільтр", "MSE", "PSNR, дБ", "SSIM"])
        for r in research_rows:
            writer.writerow([
                r["noise_type"],
                f"{r['sigma']:.2f}",
                r["filter"],
                f"{r['mse']:.2f}",
                f"{r['psnr']:.2f}",
                f"{r['ssim']:.4f}",
            ])

    # Графіки PSNR = f(sigma)
    plot_psnr_vs_sigma(
        research_rows, "гаусів", RESULTS_DIR_LAB2 / "05_psnr_vs_sigma_гаусів.png"
    )
    plot_psnr_vs_sigma(
        research_rows, "сіль-перець", RESULTS_DIR_LAB2 / "05_psnr_vs_sigma_сіль-перець.png"
    )

    # 6. Міні-дослідження: автопідбір параметрів (для гаусового та імпульсного шуму)
    print("\n[Крок 6] Міні-дослідження: автоматичний підбір параметрів фільтра...")
    tune_rng = np.random.default_rng(seed=99)
    noisy_tune_g = add_gaussian_noise(img_bgr, sigma=0.10, rng=tune_rng)
    noisy_tune_sp = add_salt_and_pepper_noise(img_bgr, amount=0.10, rng=tune_rng)

    best_cfg, tune_rows = auto_tune_filter_params(
        img_bgr,
        noisy_tune_g,
        gaussian_kernels=GAUSSIAN_KERNEL_SIZES,
        median_kernels=MEDIAN_KERNEL_SIZES,
        bilateral_diameters=BILATERAL_DIAMETERS,
    )
    best_cfg_sp, tune_rows_sp = auto_tune_filter_params(
        img_bgr,
        noisy_tune_sp,
        gaussian_kernels=GAUSSIAN_KERNEL_SIZES,
        median_kernels=MEDIAN_KERNEL_SIZES,
        bilateral_diameters=BILATERAL_DIAMETERS,
    )

    print(
        f"  * Оптимальний (гаусів, sigma=0.10): {best_cfg['filter']} "
        f"({best_cfg['param']}) -> PSNR = {float(best_cfg['psnr']):.2f} дБ, "
        f"SSIM = {float(best_cfg['ssim']):.4f}"
    )
    print(
        f"  * Оптимальний (сіль-перець, p=0.10): {best_cfg_sp['filter']} "
        f"({best_cfg_sp['param']}) -> PSNR = {float(best_cfg_sp['psnr']):.2f} дБ, "
        f"SSIM = {float(best_cfg_sp['ssim']):.4f}"
    )

    tune_csv_path = RESULTS_DIR_LAB2 / "06_auto_tune.csv"
    with open(tune_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Фільтр", "Параметр", "PSNR, дБ", "MSE", "SSIM"])
        for r in tune_rows:
            writer.writerow([
                r["filter"],
                r["param"],
                f"{float(r['psnr']):.2f}",
                f"{float(r['mse']):.2f}",
                f"{float(r['ssim']):.4f}",
            ])
        writer.writerow([
            "НАЙКРАЩИЙ",
            f"{best_cfg['filter']} ({best_cfg['param']})",
            f"{float(best_cfg['psnr']):.2f}",
            f"{float(best_cfg['mse']):.2f}",
            f"{float(best_cfg['ssim']):.4f}",
        ])

    tune_sp_csv_path = RESULTS_DIR_LAB2 / "06_auto_tune_sp.csv"
    with open(tune_sp_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Фільтр", "Параметр", "PSNR, дБ", "MSE", "SSIM"])
        for r in tune_rows_sp:
            writer.writerow([
                r["filter"],
                r["param"],
                f"{float(r['psnr']):.2f}",
                f"{float(r['mse']):.2f}",
                f"{float(r['ssim']):.4f}",
            ])
        writer.writerow([
            "НАЙКРАЩИЙ",
            f"{best_cfg_sp['filter']} ({best_cfg_sp['param']})",
            f"{float(best_cfg_sp['psnr']):.2f}",
            f"{float(best_cfg_sp['mse']):.2f}",
            f"{float(best_cfg_sp['ssim']):.4f}",
        ])

    plot_auto_tune(tune_rows, RESULTS_DIR_LAB2 / "06_auto_tune.png", sp_rows=tune_rows_sp)

    print("\n" + "=" * 70)
    print(f"[УСПІХ] Лабораторна робота №2 виконана! Усі файли у: {RESULTS_DIR_LAB2}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_lab2()
