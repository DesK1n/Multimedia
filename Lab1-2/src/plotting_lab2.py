"""Модуль побудови графіків та візуалізацій для Лабораторної роботи №2."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_noise_examples(
    original: np.ndarray,
    gaussian_noisy: np.ndarray,
    sp_noisy: np.ndarray,
    save_path: Path,
) -> None:
    """Будує порівняння оригіналу з двома типами шумів (гаусовим та імпульсним).

    Args:
        original (np.ndarray): Вихідне BGR зображення.
        gaussian_noisy (np.ndarray): Зображення з гаусовим шумом.
        sp_noisy (np.ndarray): Зображення з імпульсним шумом.
        save_path (Path): Шлях для збереження графіка.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    images = [
        (original, "Оригінальне зображення"),
        (gaussian_noisy, "Адитивний гаусів шум (sigma=0.10)"),
        (sp_noisy, "Імпульсний шум «сіль та перець» (p=5%)"),
    ]
    for ax, (img, title) in zip(axes, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Зображення прикладів шумів збережено: {save_path}")


def plot_filters_comparison(
    items: list[tuple[np.ndarray, str]], save_path: Path
) -> None:
    """Будує сітку 2x3 результатів фільтрації різними просторовими методами.

    Args:
        items (list[tuple[np.ndarray, str]]): Пари (зображення, заголовок).
        save_path (Path): Шлях для збереження графіка.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (img, title) in zip(axes.ravel(), items):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Порівняльний аналіз просторової фільтрації зображень",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік порівняння фільтрів збережено: {save_path}")


def plot_contrast_comparison(
    gray: np.ndarray, equalized: np.ndarray, clahe: np.ndarray, save_path: Path
) -> None:
    """Будує 2 рядки: верхній — зображення (Оригінал, HE, CLAHE), нижній — гістограми.

    Args:
        gray (np.ndarray): Вихідне напівтонове зображення.
        equalized (np.ndarray): Результат глобальної еквалізації гістограми.
        clahe (np.ndarray): Результат адаптивної еквалізації CLAHE.
        save_path (Path): Шлях для збереження графіка.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cols = [
        (gray, "Оригінал (напівтоновий)"),
        (equalized, "Глобальна еквалізація (HE)"),
        (clahe, "Адаптивна еквалізація (CLAHE)"),
    ]

    for j, (img, title) in enumerate(cols):
        # Зображення
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")
        axes[0, j].axis("off")

        # Гістограма
        axes[1, j].hist(
            img.ravel(), bins=256, range=(0, 255), color="#34495e", alpha=0.85
        )
        axes[1, j].set_title(f"Гістограма: {title}", fontsize=10)
        axes[1, j].set_xlim(0, 255)
        axes[1, j].grid(True, linestyle="--", alpha=0.5)
        axes[1, j].set_xlabel("Рівень яскравості [0..255]")
        axes[1, j].set_ylabel("Частота пікселів")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік контрастування та гістограм збережено: {save_path}")


def plot_thresholding_comparison(
    gray: np.ndarray,
    otsu_unif: np.ndarray,
    adapt_unif: np.ndarray,
    uneven: np.ndarray,
    otsu_uneven: np.ndarray,
    adapt_uneven: np.ndarray,
    save_path: Path,
    otsu_t_unif: float | None = None,
    otsu_t_uneven: float | None = None,
) -> None:
    """Будує порівняння бінаризації Оцу та Adaptive при рівномірному та градієнтному освітленні.

    Args:
        gray (np.ndarray): Оригінальне напівтонове зображення.
        otsu_unif (np.ndarray): Бінаризація Оцу при рівномірному освітленні.
        adapt_unif (np.ndarray): Адаптивна бінаризація при рівномірному освітленні.
        uneven (np.ndarray): Зображення з градієнтним освітленням.
        otsu_uneven (np.ndarray): Бінаризація Оцу при градієнтному освітленні.
        adapt_uneven (np.ndarray): Адаптивна бінаризація при градієнтному освітленні.
        save_path (Path): Шлях для збереження графіка.
        otsu_t_unif (float | None): Поріг Оцу для рівномірного освітлення.
        otsu_t_uneven (float | None): Поріг Оцу для градієнтного освітлення.
    """
    t_u_str = f" (T*={otsu_t_unif:.0f})" if otsu_t_unif is not None else ""
    t_g_str = f" (T*={otsu_t_uneven:.0f})" if otsu_t_uneven is not None else ""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    r1 = [
        (gray, "Рівномірне освітлення (Оригінал)"),
        (otsu_unif, f"Метод Оцу{t_u_str} (рівномірне)"),
        (adapt_unif, "Адаптивна бінаризація (рівномірне)"),
    ]
    r2 = [
        (uneven, "Нерівномірний градієнт освітлення"),
        (otsu_uneven, f"Метод Оцу{t_g_str} (колапс при градієнті)"),
        (adapt_uneven, "Адаптивна бінаризація (стійкий контур)"),
    ]

    for j, (img, title) in enumerate(r1):
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")
        axes[0, j].axis("off")

    for j, (img, title) in enumerate(r2):
        axes[1, j].imshow(img, cmap="gray")
        axes[1, j].set_title(title, fontsize=11, fontweight="bold")
        axes[1, j].axis("off")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік бінаризації збережено: {save_path}")


def plot_psnr_vs_sigma(
    rows: list[dict], noise_type: str, save_path: Path
) -> None:
    """Будує криві залежності PSNR = f(sigma) для різних фільтрів.

    Args:
        rows (list[dict]): Таблиця результатів експерименту.
        noise_type (str): Назва типу шуму ('гаусів' або 'сіль-перець').
        save_path (Path): Шлях для збереження графіка.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    filters = ["без фільтра", "Gaussian", "Median", "Bilateral"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    markers = ["x", "o", "s", "^"]

    for f_name, col, mrk in zip(filters, colors, markers):
        subset = [r for r in rows if r["noise_type"] == noise_type and r["filter"] == f_name]
        xs = [r["sigma"] for r in subset]
        ys = [r["psnr"] for r in subset]
        ax.plot(
            xs,
            ys,
            label=f_name,
            color=col,
            marker=mrk,
            linewidth=2,
            markersize=6,
        )

    ax.set_title(f"Залежність PSNR = f(sigma) для шуму: {noise_type}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Рівень шуму (sigma / частка імпульсів p)", fontsize=11)
    ax.set_ylabel("PSNR, дБ", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік PSNR=f(sigma) збережено: {save_path}")


def _plot_tune_row(
    ax_k: plt.Axes,
    ax_d: plt.Axes,
    rows: list[dict],
    noise_label: str,
) -> None:
    """Відображає криві автопідбору ядра (Gaussian/Median) та діаметра (Bilateral)."""
    styles = {
        "Gaussian": ("#1f77b4", "o"),
        "Median": ("#2ca02c", "s"),
    }
    k_vals_all: list[float] = []
    for f_name, (col, mrk) in styles.items():
        subset = [r for r in rows if r["filter"] == f_name]
        xs = [int(str(r["param"]).split("=")[1]) for r in subset]
        ys = [float(r["psnr"]) for r in subset]
        k_vals_all.extend(ys)
        ax_k.plot(xs, ys, label=f_name, color=col, marker=mrk, linewidth=2, markersize=6)
        for x, y in zip(xs, ys):
            ax_k.annotate(
                f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8
            )

    best_k_row = max(
        (r for r in rows if r["filter"] in ("Gaussian", "Median")),
        key=lambda r: float(r["psnr"]),
    )
    ax_k.set_title(
        f"{noise_label}: вплив розміру ядра k\n"
        f"(найкраще: {best_k_row['filter']} {best_k_row['param']}, {float(best_k_row['psnr']):.2f} дБ)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_k.set_xlabel("Розмір ядра k (пікселі)", fontsize=10)
    ax_k.set_ylabel("PSNR, дБ", fontsize=10)
    ax_k.set_xticks([3, 5, 7, 9, 11])
    if k_vals_all:
        ax_k.set_ylim(min(k_vals_all) - 1.0, max(k_vals_all) + 2.0)
    ax_k.grid(True, linestyle="--", alpha=0.6)
    ax_k.legend(fontsize=9)

    b_sub = [r for r in rows if r["filter"] == "Bilateral"]
    bx = [int(str(r["param"]).split("=")[1]) for r in b_sub]
    by = [float(r["psnr"]) for r in b_sub]
    ax_d.plot(bx, by, label="Bilateral", color="#d62728", marker="^", linewidth=2, markersize=7)
    for x, y in zip(bx, by):
        ax_d.annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5
        )
    best_b = max(b_sub, key=lambda r: float(r["psnr"]))
    ax_d.set_title(
        f"{noise_label}: Bilateral за діаметром d\n"
        f"(найкраще: {best_b['param']}, {float(best_b['psnr']):.2f} дБ)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_d.set_xlabel("Діаметр околу d (пікселі)", fontsize=10)
    ax_d.set_ylabel("PSNR, дБ", fontsize=10)
    ax_d.set_xticks(bx)
    if by:
        span = max(max(by) - min(by), 1.0)
        ax_d.set_ylim(min(by) - 0.35 * span, max(by) + 0.45 * span)
    ax_d.grid(True, linestyle="--", alpha=0.6)
    ax_d.legend(fontsize=9)


def plot_auto_tune(
    rows: list[dict],
    save_path: Path,
    sp_rows: list[dict] | None = None,
) -> None:
    """Будує графік підбору параметрів фільтрації за критерієм PSNR.

    Args:
        rows (list[dict]): Результати сіткового пошуку для гаусового шуму.
        save_path (Path): Шлях для збереження графіка.
        sp_rows (list[dict] | None): Результати сіткового пошуку для імпульсного шуму.
    """
    if sp_rows:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        _plot_tune_row(axes[0, 0], axes[0, 1], rows, "Гаусів шум (sigma=0.10)")
        _plot_tune_row(axes[1, 0], axes[1, 1], sp_rows, "Шум «сіль-перець» (p=0.10)")
        fig.suptitle(
            "Автоматичний підбір параметрів фільтрації та вплив розміру апертури",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        _plot_tune_row(axes[0], axes[1], rows, "Гаусів шум (sigma=0.10)")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік автопідбору збережено: {save_path}")
