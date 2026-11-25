"""Модуль амплітудного квантування звукових сигналів.

Реалізує рівномірне симетричне квантування для заданої розрядності (бітової
глибини) та розрахунок шуму квантування (Quantization Error).
"""

import numpy as np


def quantize_uniform(x: np.ndarray, bit_depth: int) -> np.ndarray:
    r"""Виконує рівномірне квантування амплітуди відліків у діапазоні [-1.0, 1.0].

    Кількість доступних рівнів сітки становить рівно $L = 2^b$.
    Нормоване відображення:
    $$q_{\text{int}} = \text{clip}\left(\text{round}\left(\frac{x + 1.0}{2.0} \cdot (L - 1)\right), 0, L - 1\right)$$
    $$\hat{x} = \frac{q_{\text{int}}}{L - 1} \cdot 2.0 - 1.0$$

    Args:
        x: Вхідний масив відліків у діапазоні [-1.0, 1.0].
        bit_depth: Бітова глибина квантування (b біт >= 1).

    Returns:
        np.ndarray: Квантований масив відліків (float64) з рівно 2^b рівнями.

    Raises:
        ValueError: Якщо bit_depth < 1.
    """
    if bit_depth < 1:
        raise ValueError(f"Бітова глибина має бути >= 1, отримано {bit_depth}")
    levels = 2**bit_depth
    # Нормалізуємо з [-1, 1] до [0, levels - 1]
    scaled = (x + 1.0) / 2.0 * float(levels - 1)
    q_int = np.clip(np.round(scaled), 0, levels - 1)
    # Повертаємо до діапазону [-1, 1]
    q_norm = (q_int / float(levels - 1)) * 2.0 - 1.0
    return q_norm.astype(np.float64)


def compute_quantization_error(
    original: np.ndarray, quantized: np.ndarray
) -> np.ndarray:
    r"""Обчислює миттєвий сигнал похибки (шуму) квантування:
    $$e[n] = \hat{x}[n] - x[n]$$

    Args:
        original: Первинний аналоговий сигнал.
        quantized: Квантований сигнал.

    Returns:
        np.ndarray: Масив відліків похибки.
    """
    return quantized - original
