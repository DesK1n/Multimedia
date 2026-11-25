"""Модуль дискретизації та зміни частоти відліків (Resampling).

Реалізує поліфазну фільтрацію із захистом від накладання спектрів (Anti-Aliasing)
відповідно до теореми Найквіста–Котельникова–Шеннона.
"""

from math import gcd

import numpy as np
from scipy import signal


def resample_poly_filter(
    x: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Виконує повторну дискретизацію сигналу з поліфазною фільтрацією.

    Застосовує низькочастотний антиаліасинговий фільтр перед проріджуванням
    сигналу для усунення ефекту накладання спектральних дзеркальних частот.

    Args:
        x: Вхідний масив відліків.
        orig_sr: Початкова частота дискретизації (Гц).
        target_sr: Цільова частота дискретизації (Гц).

    Returns:
        np.ndarray: Ресемпльований масив відліків.
    """
    if orig_sr == target_sr:
        return x.copy()

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    return signal.resample_poly(x, up, down).astype(np.float64)


def resample_roundtrip(
    x: np.ndarray, orig_sr: int, intermediate_sr: int
) -> np.ndarray:
    """Виконує цикл ресемплінгу (пониження до intermediate_sr і повернення назад).

    Дозволяє оцінити величину незворотних інформаційних втрат високочастотних
    складових сигналу при зниженні частоти дискретизації нижче $2 f_{max}$.

    Args:
        x: Оригінальний масив відліків на частоті orig_sr.
        orig_sr: Базова частота дискретизації (Гц).
        intermediate_sr: Проміжна дослідницька частота (Гц).

    Returns:
        np.ndarray: Відновлений сигнал довжини len(x) на базовій частоті orig_sr.
    """
    if intermediate_sr == orig_sr:
        return x.copy()

    # Зниження частоти (Decimation + Low-pass filtering)
    downsampled = resample_poly_filter(x, orig_sr, intermediate_sr)
    # Повернення назад до базової частоти
    reconstructed = resample_poly_filter(downsampled, intermediate_sr, orig_sr)

    # Узгодження довжини відліків (компенсація округлення)
    target_len = len(x)
    if len(reconstructed) > target_len:
        reconstructed = reconstructed[:target_len]
    elif len(reconstructed) < target_len:
        reconstructed = np.pad(
            reconstructed, (0, target_len - len(reconstructed))
        )

    return reconstructed
