"""Модуль побудови та збереження графіків і візуальних матеріалів для звіту."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_color_channels(channels_dict: dict[str, np.ndarray], save_path: Path) -> None:
    """Будує та зберігає сітку 3x3 напівтонових зображень каналів RGB, HSV, YCbCr.

    Args:
        channels_dict (dict[str, np.ndarray]): Словник матриць каналів.
        save_path (Path): Шлях для збереження рисунка.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    items = [
        (channels_dict["r"], "Канал R (Червоний)"),
        (channels_dict["g"], "Канал G (Зелений)"),
        (channels_dict["b"], "Канал B (Синій)"),
        (channels_dict["h"], "Канал H (Відтінок, Hue)"),
        (channels_dict["s"], "Канал S (Насиченість, Saturation)"),
        (channels_dict["v"], "Канал V (Яскравість, Value)"),
        (channels_dict["y"], "Канал Y (Яскравість, Luma)"),
        (channels_dict["cr"], "Канал Cr (Червона колірність)"),
        (channels_dict["cb"], "Канал Cb (Синя колірність)"),
    ]

    for ax, (ch, title) in zip(axes.ravel(), items):
        ax.imshow(ch, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Декомпозиція на колірні канали: моделі RGB, HSV, YCbCr",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік каналів збережено: {save_path}")


def plot_grayscale_comparison(
    gray_manual: np.ndarray,
    gray_cv2: np.ndarray,
    diff: np.ndarray,
    save_path: Path,
) -> None:
    """Будує порівняння ручного та бібліотечного перетворення у Grayscale.

    Args:
        gray_manual (np.ndarray): Ручна формула (0.299R + 0.587G + 0.114B).
        gray_cv2 (np.ndarray): cv2.cvtColor(BGR2GRAY).
        diff (np.ndarray): Матриця абсолютної різниці.
        save_path (Path): Шлях для збереження рисунка.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(gray_manual, cmap="gray")
    axes[0].set_title("Ручна формула: 0.299R+0.587G+0.114B", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gray_cv2, cmap="gray")
    axes[1].set_title("Бібліотечне: cv2.cvtColor", fontsize=11)
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title(
        f"Абсолютна різниця (макс={int(diff.max())}, сер={diff.mean():.4f})",
        fontsize=11,
    )
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік порівняння Grayscale збережено: {save_path}")


def plot_geometry_transforms(
    transforms: dict[str, np.ndarray], save_path: Path
) -> None:
    """Будує панель геометричних перетворень (оригінал, масштаб, поворот, кроп).

    Args:
        transforms (dict[str, np.ndarray]): Словник зображень після трансформацій.
        save_path (Path): Шлях для збереження рисунка.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (transforms["original"], "Оригінал (512x512)"),
        (transforms["scaled_down"], "Масштабування 0.5x"),
        (transforms["rotated"], "Поворот на 30°"),
        (transforms["cropped"], "Кадрування центру (60%)"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік геометричних перетворень збережено: {save_path}")


def plot_interpolation_comparison(
    interp_results: dict[str, dict[str, float | np.ndarray]], save_path: Path
) -> None:
    """Будує візуальне порівняння методів інтерполяції після циклу зменшення/відновлення.

    Args:
        interp_results (dict[str, dict[str, float | np.ndarray]]): Результати інтерполяцій.
        save_path (Path): Шлях для збереження рисунка.
    """
    fig, axes = plt.subplots(1, len(interp_results), figsize=(16, 4))

    for ax, (name, data) in zip(axes, interp_results.items()):
        img_rgb = cv2.cvtColor(data["restored_img"], cv2.COLOR_BGR2RGB)
        psnr_val = float(data["psnr"])
        ax.imshow(img_rgb)
        ax.set_title(f"{name}\nPSNR = {psnr_val:.2f} дБ", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Порівняння методів інтерполяції після циклу зменшення в 4 рази та відновлення",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік порівняння інтерполяцій збережено: {save_path}")


def plot_metrics_validation(
    noisy_img: np.ndarray, ssim_map: np.ndarray, save_path: Path
) -> None:
    """Будує зображення зі штучним шумом та відповідну просторову карту SSIM.

    Args:
        noisy_img (np.ndarray): Зображення з доданим гаусовим шумом.
        ssim_map (np.ndarray): Просторова карта похибки SSIM.
        save_path (Path): Шлях для збереження рисунка.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].imshow(noisy_img, cmap="gray")
    axes[0].set_title("Зображення зі штучним гаусовим шумом", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(ssim_map, cmap="viridis")
    axes[1].set_title("Просторова карта подібності SSIM", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік валідації метрик збережено: {save_path}")


def plot_jpeg_experiment(
    jpeg_rows: list[dict[str, float]], save_path: Path
) -> None:
    """Будує графіки залежностей PSNR = f(Q) та SSIM = f(Q).

    Args:
        jpeg_rows (list[dict[str, float]]): Результати компресії для кожного Q.
        save_path (Path): Шлях для збереження рисунка.
    """
    qs = [int(r["Q"]) for r in jpeg_rows]
    psnrs = [r["psnr"] for r in jpeg_rows]
    ssims = [r["ssim"] for r in jpeg_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # PSNR = f(Q)
    axes[0].plot(qs, psnrs, marker="o", color="#1f77b4", linewidth=2, markersize=6)
    axes[0].set_title("Залежність PSNR = f(Q)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Коефіцієнт якості Q", fontsize=11)
    axes[0].set_ylabel("PSNR, дБ", fontsize=11)
    axes[0].set_ylim(min(psnrs) - 1.5, max(psnrs) + 2.8)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    for q, p in zip(qs, psnrs):
        axes[0].annotate(
            f"{p:.1f}", (q, p), textcoords="offset points", xytext=(0, 7), ha="center"
        )

    # SSIM = f(Q)
    axes[1].plot(qs, ssims, marker="s", color="#ff7f0e", linewidth=2, markersize=6)
    axes[1].set_title("Залежність SSIM = f(Q)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Коефіцієнт якості Q", fontsize=11)
    axes[1].set_ylabel("SSIM", fontsize=11)
    axes[1].set_ylim(min(ssims) - 0.03, min(1.03, max(ssims) + 0.045))
    axes[1].grid(True, linestyle="--", alpha=0.6)
    for q, s in zip(qs, ssims):
        axes[1].annotate(
            f"{s:.3f}", (q, s), textcoords="offset points", xytext=(0, 7), ha="center"
        )

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік залежностей компресії збережено: {save_path}")


def plot_jpeg_artifacts(
    img_bgr: np.ndarray, low_q_bgr: np.ndarray, save_path: Path
) -> None:
    """Будує 2x2 візуалізацію блокових артефактів (Q=10), карти похибки та збільшеного 64x64 фрагмента.

    Args:
        img_bgr (np.ndarray): Оригінальне BGR зображення.
        low_q_bgr (np.ndarray): Відновлене зображення з низькою якістю Q=10.
        save_path (Path): Шлях для збереження рисунка.
    """
    diff = cv2.absdiff(img_bgr, low_q_bgr)
    diff_amplified = np.clip(diff.astype(np.int32) * 5, 0, 255).astype(np.uint8)

    h, w = img_bgr.shape[:2]
    y0, x0 = min(180, max(0, h - 64)), min(210, max(0, w - 64))
    y1, x1 = y0 + 64, x0 + 64

    low_q_with_roi = low_q_bgr.copy()
    cv2.rectangle(low_q_with_roi, (x0, y0), (x1, y1), (0, 255, 255), 2)

    crop_orig = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    crop_low = cv2.cvtColor(low_q_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    axes[0, 0].imshow(cv2.cvtColor(low_q_with_roi, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Стиснене зображення (JPEG, Q=10) + зона ROI", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Посилена різниця |Оригінал - Стиснене| (x5)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(crop_orig, interpolation="nearest")
    axes[1, 0].set_title("Фрагмент 64x64 до стиснення (Оригінал)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(crop_low, interpolation="nearest")
    axes[1, 1].set_title("Фрагмент 64x64 після JPEG Q=10 (блоки 8x8)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік артефактів компресії збережено: {save_path}")
