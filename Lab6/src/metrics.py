"""Модуль розрахунку метрик якості та відношення сигнал/шум (SNR).

Містить методи експериментального розрахунку Signal-to-Noise Ratio (SNR) у дБ
та теоретичну формулу для гармонічного квантованого сигналу.
"""

import numpy as np
from scipy import signal


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    r"""Обчислює відношення сигнал/шум (Signal-to-Noise Ratio) у децибелах.

    Формула розрахунку:
    $$SNR = 10 \cdot \log_{10} \left( \frac{\sum_{n=0}^{N-1} x^2[n]}{\sum_{n=0}^{N-1} (x[n] - \hat{x}[n])^2} \right) \quad [\text{дБ}]$$

    Args:
        original: Еталонний масив відліків вихідного сигналу.
        processed: Спотворений масив відліків (після квантування або ресемплінгу).

    Returns:
        float: Значення SNR у децибелах (дБ). При повній відсутності шуму повертає inf.
    """
    orig = np.asarray(original, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)

    if len(proc) != len(orig):
        proc = signal.resample(proc, len(orig))

    power_signal = float(np.mean(orig**2))
    power_noise = float(np.mean((orig - proc) ** 2))

    if power_noise <= 1e-18:
        return float("inf")

    snr_db = 10.0 * np.log10(power_signal / power_noise)
    return float(snr_db)


def theoretical_snr(bit_depth: int) -> float:
    r"""Обчислює теоретичну межу SNR для повномасштабного гармонічного сигналу:
    $$SNR_{\text{dB}} \approx 6.02 \cdot b + 1.76 \quad [\text{дБ}]$$

    Args:
        bit_depth: Розрядність шкали квантування (b біт).

    Returns:
        float: Теоретичне значення SNR у дБ.
    """
    return float(6.02 * bit_depth + 1.76)
