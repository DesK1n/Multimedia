"""Модуль реалізації та верифікації метрик якості цифрових зображень.

Містить власноручні реалізації на NumPy та порівняння з еталонними функціями
бібліотеки scikit-image:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
"""

import cv2
import numpy as np
from skimage.metrics import mean_squared_error as sk_mse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def mse_manual(img1: np.ndarray, img2: np.ndarray) -> float:
    r"""Обчислює середньоквадратичну помилку (MSE) між двома зображеннями.

    Формула:
    $$MSE = \frac{1}{M \cdot N} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I(i, j) - K(i, j)]^2$$

    Args:
        img1 (np.ndarray): Перше зображення (еталон).
        img2 (np.ndarray): Друге зображення (спотворене або відновлене).

    Returns:
        float: Значення середньоквадратичної похибки.

    Raises:
        ValueError: Якщо зображення мають різну розмірність або форму.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Розмірності зображень не збігаються: {img1.shape} != {img2.shape}"
        )
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def rmse_manual(img1: np.ndarray, img2: np.ndarray) -> float:
    r"""Обчислює корінь із середньоквадратичної помилки (RMSE).

    Формула:
    $$RMSE = \sqrt{MSE}$$

    Args:
        img1 (np.ndarray): Перше зображення.
        img2 (np.ndarray): Друге зображення.

    Returns:
        float: Значення кореневої середньоквадратичної похибки.
    """
    return float(np.sqrt(mse_manual(img1, img2)))


def psnr_manual(
    img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0
) -> float:
    r"""Обчислює пікове відношення сигналу до шуму (PSNR) у децибелах (дБ).

    Формула:
    $$PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)$$

    Args:
        img1 (np.ndarray): Перше зображення (еталон).
        img2 (np.ndarray): Друге зображення (тестове).
        max_val (float): Максимально можливе значення відліку (за замовчуванням 255.0).

    Returns:
        float: Значення PSNR у дБ (при MSE=0 повертає float('inf')).
    """
    mse_val = mse_manual(img1, img2)
    if mse_val == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse_val))


def ssim_manual_gray(
    img1: np.ndarray, img2: np.ndarray, win_size: int = 7
) -> float:
    r"""Обчислює індекс структурної подібності (SSIM) для напівтонових зображень.

    Формула Ванга та ін. (Wang et al., 2004):
    $$SSIM(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
    де локальні математичні сподівання та дисперсії обчислюються у ковзному
    вікні розміром win_size x win_size за допомогою box-фільтра з незміщеною
    оцінкою вибіркової дисперсії $N / (N - 1)$.

    Args:
        img1 (np.ndarray): Еталонне одноканальне (Grayscale) зображення uint8 або float.
        img2 (np.ndarray): Порівнюване одноканальне зображення.
        win_size (int): Розмір ковзного вікна оцінки локальних статистик.

    Returns:
        float: Середній індекс структурної подібності на інтервалі [-1, 1].

    Raises:
        ValueError: Якщо зображення мають більше одного каналу або різні розміри.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Розмірності зображень не збігаються: {img1.shape} != {img2.shape}"
        )
    if img1.ndim != 2:
        raise ValueError(
            f"Функція розрахована на 2D Grayscale зображення, отримано shape={img1.shape}"
        )

    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    def local_mean(mat: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            mat,
            ddepth=-1,
            ksize=(win_size, win_size),
            borderType=cv2.BORDER_REPLICATE,
        )

    mu_x = local_mean(x)
    mu_y = local_mean(y)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    n_pixels = win_size ** 2
    cov_norm = n_pixels / (n_pixels - 1) if n_pixels > 1 else 1.0

    sigma_x2 = cov_norm * (local_mean(x * x) - mu_x2)
    sigma_y2 = cov_norm * (local_mean(y * y) - mu_y2)
    sigma_xy = cov_norm * (local_mean(x * y) - mu_xy)

    l_dynamic_range = 255.0
    c1 = (0.01 * l_dynamic_range) ** 2
    c2 = (0.03 * l_dynamic_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator

    pad = win_size // 2
    if pad > 0 and ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]

    return float(np.mean(ssim_map))


def compare_manual_vs_skimage(
    img_ref: np.ndarray, img_distorted: np.ndarray
) -> dict[str, dict[str, float]]:
    """Звіряє ручні реалізації метрик з еталонними функціями бібліотеки scikit-image.

    Args:
        img_ref (np.ndarray): Еталонне зображення (Grayscale).
        img_distorted (np.ndarray): Спотворене зображення.

    Returns:
        dict[str, dict[str, float]]: Словник порівняння значень метрик та абсолютних різниць.
    """
    mse_own = mse_manual(img_ref, img_distorted)
    rmse_own = rmse_manual(img_ref, img_distorted)
    psnr_own = psnr_manual(img_ref, img_distorted)
    ssim_own = ssim_manual_gray(img_ref, img_distorted)

    mse_lib = float(sk_mse(img_ref, img_distorted))
    rmse_lib = float(np.sqrt(mse_lib))
    psnr_lib = float(sk_psnr(img_ref, img_distorted, data_range=255))
    ssim_lib = float(sk_ssim(img_ref, img_distorted, data_range=255))

    return {
        "MSE": {
            "manual": mse_own,
            "library": mse_lib,
            "abs_diff": abs(mse_own - mse_lib),
        },
        "RMSE": {
            "manual": rmse_own,
            "library": rmse_lib,
            "abs_diff": abs(rmse_own - rmse_lib),
        },
        "PSNR": {
            "manual": psnr_own,
            "library": psnr_lib,
            "abs_diff": abs(psnr_own - psnr_lib),
        },
        "SSIM": {
            "manual": ssim_own,
            "library": ssim_lib,
            "abs_diff": abs(ssim_own - ssim_lib),
        },
    }
