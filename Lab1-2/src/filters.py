"""Модуль просторової фільтрації (Gaussian, Median, Bilateral) та автопідбору параметрів."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

from src.metrics import mse_manual, psnr_manual


def apply_gaussian_filter(
    img: np.ndarray, ksize: int = 5, sigma: float = 1.5
) -> np.ndarray:
    r"""Застосовує лінійний фільтр Гауса (Gaussian Blur).

    Ядро фільтра:
    $$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

    Args:
        img (np.ndarray): Вхідне цифрове зображення (uint8).
        ksize (int): Розмір квадратної апертури ядра (має бути непарним числом).
        sigma (float): Стандартне відхилення гаусового розподілу за обома осями.

    Returns:
        np.ndarray: Відфільтроване зображення.
    """
    if ksize % 2 == 0:
        raise ValueError(f"Розмір ядра має бути непарним, отримано ksize={ksize}")
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Застосовує нелінійний ранговий медіанний фільтр (Median Filter).

    Кожен відлік замінюється на медіану варіаційного ряду значень у вікні ksize x ksize.
    Ідеально підходить для усунення імпульсних викидів (шум «сіль та перець»).

    Args:
        img (np.ndarray): Вхідне цифрове зображення.
        ksize (int): Розмір вікна (непарне додатне число).

    Returns:
        np.ndarray: Відфільтроване зображення.
    """
    if ksize % 2 == 0:
        raise ValueError(f"Розмір вікна має бути непарним, отримано ksize={ksize}")
    return cv2.medianBlur(img, ksize)


def apply_bilateral_filter(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    r"""Застосовує двосторонній білатеральний фільтр із збереженням контурів.

    Математична модель:
    $$I_{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \cdot G_{\sigma_s}(\|p - q\|) \cdot G_{\sigma_r}(|I(p) - I(q)|)$$

    Args:
        img (np.ndarray): Вхідне зображення.
        d (int): Діаметр околу кожного пікселя при фільтрації.
        sigma_color (float): Фільтр у колірному просторі (ваги різниці яскравостей).
        sigma_space (float): Фільтр у просторових координатах (ваги відстаней).

    Returns:
        np.ndarray: Відфільтроване зображення зі збереженими різкими краями.
    """
    return cv2.bilateralFilter(
        img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )


def auto_tune_filter_params(
    clean_img: np.ndarray,
    noisy_img: np.ndarray,
    gaussian_kernels: list[int] | None = None,
    median_kernels: list[int] | None = None,
    bilateral_diameters: list[int] | None = None,
) -> tuple[dict[str, str | float], list[dict[str, str | float]]]:
    r"""Здійснює сітковий перебір та автоматичний підбір оптимальних параметрів фільтрів.

    Критерій оптимізації — максимізація показника PSNR відносно недеградованого еталону:
    $$\theta^* = \arg\max_\theta PSNR(I_{\text{clean}}, f_\theta(I_{\text{noisy}}))$$

    Args:
        clean_img (np.ndarray): Недеградований еталон (оригінал).
        noisy_img (np.ndarray): Зашумлене зображення.
        gaussian_kernels (list[int] | None): Список розмірів ядер для Гауса.
        median_kernels (list[int] | None): Список розмірів ядер для медіани.
        bilateral_diameters (list[int] | None): Список діаметрів для білатерального фільтра.

    Returns:
        tuple[dict[str, str | float], list[dict[str, str | float]]]:
            - Словник найкращого налаштування (фільтр, параметр, PSNR);
            - Повний список результатів для побудови графіків та експорту таблиці.
    """
    if gaussian_kernels is None:
        gaussian_kernels = [3, 5, 7, 9, 11]
    if median_kernels is None:
        median_kernels = [3, 5, 7, 9, 11]
    if bilateral_diameters is None:
        bilateral_diameters = [5, 9, 13, 17]

    rows: list[dict[str, str | float]] = []
    best_config: dict[str, str | float] = {
        "filter": "None",
        "param": "None",
        "psnr": -float("inf"),
        "mse": float("inf"),
        "ssim": 0.0,
    }

    def _compute_ssim(ref: np.ndarray, target: np.ndarray) -> float:
        if ref.ndim == 3:
            return float(sk_ssim(ref, target, channel_axis=2, data_range=255))
        return float(sk_ssim(ref, target, data_range=255))

    # 1. Gaussian filter sweep
    for k in gaussian_kernels:
        filtered = apply_gaussian_filter(noisy_img, ksize=k, sigma=0.0)
        p_val = psnr_manual(clean_img, filtered)
        m_val = mse_manual(clean_img, filtered)
        s_val = _compute_ssim(clean_img, filtered)
        param_label = f"kernel={k}"
        rows.append(
            {
                "filter": "Gaussian",
                "param": param_label,
                "psnr": p_val,
                "mse": m_val,
                "ssim": s_val,
            }
        )
        if p_val > float(best_config["psnr"]):
            best_config = {
                "filter": "Gaussian",
                "param": param_label,
                "psnr": p_val,
                "mse": m_val,
                "ssim": s_val,
            }

    # 2. Median filter sweep
    for k in median_kernels:
        filtered = apply_median_filter(noisy_img, ksize=k)
        p_val = psnr_manual(clean_img, filtered)
        m_val = mse_manual(clean_img, filtered)
        s_val = _compute_ssim(clean_img, filtered)
        param_label = f"kernel={k}"
        rows.append(
            {
                "filter": "Median",
                "param": param_label,
                "psnr": p_val,
                "mse": m_val,
                "ssim": s_val,
            }
        )
        if p_val > float(best_config["psnr"]):
            best_config = {
                "filter": "Median",
                "param": param_label,
                "psnr": p_val,
                "mse": m_val,
                "ssim": s_val,
            }

    # 3. Bilateral filter sweep
    for d in bilateral_diameters:
        filtered = apply_bilateral_filter(
            noisy_img, d=d, sigma_color=75.0, sigma_space=75.0
        )
        p_val = psnr_manual(clean_img, filtered)
        m_val = mse_manual(clean_img, filtered)
        s_val = _compute_ssim(clean_img, filtered)
        param_label = f"d={d}"
        rows.append(
            {
                "filter": "Bilateral",
                "param": param_label,
                "psnr": p_val,
                "mse": m_val,
                "ssim": s_val,
            }
        )
        if p_val > float(best_config["psnr"]):
            best_config = {
                "filter": "Bilateral",
                "param": param_label,
                "psnr": p_val,
                "mse": m_val,
                "ssim": s_val,
            }

    return best_config, rows
