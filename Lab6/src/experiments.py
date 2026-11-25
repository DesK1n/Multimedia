"""Модуль проведення експериментальних досліджень дискретизації та квантування.

Реалізує конвеєри досліджень:
- Ресемплінг на множину частот RESAMPLE_RATES.
- Квантування на розрядності QUANT_BIT_DEPTHS.
- Розрахунок кривих SNR = f(b) та SNR = f(fs).
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from src import metrics, quantization, sampling


def run_resampling_analysis(
    x: np.ndarray, orig_sr: int, rates: List[int]
) -> Dict[int, np.ndarray]:
    """Виконує повторне дискретизування сигналу для списку цільових частот.

    Args:
        x: Вихідний масив відліків.
        orig_sr: Базова частота дискретизації (Гц).
        rates: Список частот для ресемплінгу.

    Returns:
        Dict[int, np.ndarray]: Словник {частота: ресемпльований_сигнал}.
    """
    results: Dict[int, np.ndarray] = {}
    for sr in rates:
        resampled_sig = sampling.resample_poly_filter(x, orig_sr, sr)
        results[sr] = resampled_sig
        print(f"  Частота: {sr:5d} Гц -> {len(resampled_sig):6d} відліків")
    return results


def run_quantization_analysis(
    x: np.ndarray, bit_depths: List[int]
) -> Dict[int, np.ndarray]:
    """Виконує амплітудне квантування сигналу для списку бітових глибин.

    Args:
        x: Вихідний нормалізований сигнал.
        bit_depths: Список розрядностей (біт).

    Returns:
        Dict[int, np.ndarray]: Словник {розрядність: квантований_сигнал}.
    """
    results: Dict[int, np.ndarray] = {}
    for b in bit_depths:
        quantized_sig = quantization.quantize_uniform(x, b)
        results[b] = quantized_sig
        levels = 2**b
        snr = metrics.compute_snr(x, quantized_sig)
        print(
            f"  Розрядність: {b:2d} біт ({levels:5d} рівнів) -> SNR = {snr:6.2f} дБ"
        )
    return results


def run_snr_bit_depth_curve(
    x: np.ndarray, bit_depths: List[int]
) -> Tuple[List[int], List[float], List[float]]:
    """Розраховує залежність SNR від бітової глибини квантування.

    Args:
        x: Еталонний сигнал.
        bit_depths: Діапазон досліджуваних розрядностей.

    Returns:
        Tuple[List[int], List[float], List[float]]: (біти, практичний_SNR, теоретичний_SNR).
    """
    snr_practical: List[float] = []
    snr_theory: List[float] = []

    for b in bit_depths:
        quantized_sig = quantization.quantize_uniform(x, b)
        snr_p = metrics.compute_snr(x, quantized_sig)
        snr_t = metrics.theoretical_snr(b)
        snr_practical.append(round(snr_p, 2))
        snr_theory.append(round(snr_t, 2))

    return bit_depths, snr_practical, snr_theory


def run_snr_sample_rate_curve(
    x: np.ndarray, orig_sr: int, rates: List[int]
) -> Tuple[List[int], List[float]]:
    """Розраховує залежність SNR від частоти дискретизації за схемою round-trip.

    Args:
        x: Еталонний сигнал.
        orig_sr: Базова частота дискретизації.
        rates: Список проміжних частот.

    Returns:
        Tuple[List[int], List[float]]: (частоти, значення_SNR).
    """
    snr_values: List[float] = []

    for sr in rates:
        reconstructed = sampling.resample_roundtrip(x, orig_sr, sr)
        snr = metrics.compute_snr(x, reconstructed)
        snr_values.append(round(snr, 2))

    return rates, snr_values


def run_aliasing_demo(
    orig_sr: int = 44100,
    decimation_factor: int = 5,
    low_freq: float = 1000.0,
    high_freq: float = 5000.0,
) -> Dict[str, Any]:
    """Демонструє ефект накладання спектрів (Aliasing) та роботу антиаліасингового фільтра.

    Генерує двочастотний сигнал (1000 Гц + 5000 Гц) при orig_sr=44100 Гц і знижує
    частоту в 5 разів до target_sr=8820 Гц (f_Nyquist = 4410 Гц):
    - наївна децимація (без фільтра) породжує фальшиву частоту |5000 - 8820| = 3820 Гц;
    - поліфазний ресемплінг повністю придушує 5000 Гц до зниження частоти.

    Args:
        orig_sr: Початкова частота дискретизації (Гц).
        decimation_factor: Коефіцієнт проріджування відліків.
        low_freq: Дозволена низька частота (< target_sr / 2).
        high_freq: Заборонена висока частота (> target_sr / 2).

    Returns:
        Dict[str, Any]: Спектри та кількісні характеристики аліасингу.
    """
    duration = 1.0
    t = np.arange(int(orig_sr * duration), dtype=np.float64) / float(orig_sr)
    sig = 0.7 * np.sin(2.0 * np.pi * low_freq * t) + 0.6 * np.sin(
        2.0 * np.pi * high_freq * t
    )

    target_sr = orig_sr // decimation_factor
    nyquist_target = target_sr / 2.0
    expected_alias = abs(high_freq - target_sr)

    naive_decimated = sig[::decimation_factor]
    filtered_decimated = sampling.resample_poly_filter(sig, orig_sr, target_sr)

    def _norm_fft_db(arr: np.ndarray, sr_val: int) -> Tuple[np.ndarray, np.ndarray]:
        win = np.hanning(len(arr))
        spec = np.abs(np.fft.rfft(arr * win))
        spec_norm = spec / (np.max(spec) + 1e-12)
        spec_db = 20.0 * np.log10(np.maximum(spec_norm, 1e-6))
        freqs = np.fft.rfftfreq(len(arr), d=1.0 / float(sr_val))
        return freqs, spec_db

    f_orig, db_orig = _norm_fft_db(sig, orig_sr)
    f_naive, db_naive = _norm_fft_db(naive_decimated, target_sr)
    f_filt, db_filt = _norm_fft_db(filtered_decimated, target_sr)

    alias_window = (f_naive >= expected_alias - 50.0) & (
        f_naive <= expected_alias + 50.0
    )
    naive_alias_idx = int(np.argmax(db_naive[alias_window]))
    measured_alias_freq = float(f_naive[alias_window][naive_alias_idx])
    naive_alias_db = float(db_naive[alias_window][naive_alias_idx])

    filt_window = (f_filt >= expected_alias - 50.0) & (
        f_filt <= expected_alias + 50.0
    )
    filtered_alias_db = float(np.max(db_filt[filt_window]))

    return {
        "orig_sr": orig_sr,
        "target_sr": target_sr,
        "nyquist_target": nyquist_target,
        "low_freq": low_freq,
        "high_freq": high_freq,
        "expected_alias_freq": expected_alias,
        "measured_alias_freq": round(measured_alias_freq, 1),
        "naive_alias_db": round(naive_alias_db, 2),
        "filtered_alias_db": round(filtered_alias_db, 2),
        "suppression_db": round(naive_alias_db - filtered_alias_db, 2),
        "f_orig": f_orig,
        "db_orig": db_orig,
        "f_naive": f_naive,
        "db_naive": db_naive,
        "f_filt": f_filt,
        "db_filt": db_filt,
    }
