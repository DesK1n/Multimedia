"""Модуль генерації та введення-виведення звукових сигналів.

Забезпечує математичний синтез полігармонічного широкосмугового сигналу,
збереження у форматі WAV (PCM 16-bit) та зчитування метаданих аудіофайлу.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf

from src import config


def create_harmonic_signal(
    sample_rate: int = config.BASE_SAMPLE_RATE,
    duration: float = config.DURATION_SEC,
    f0: float = config.BASE_FREQ,
) -> np.ndarray:
    r"""Генерує широкосмуговий полігармонічний сигнал із основним тоном та високочастотними обертонами.

    Компоненти сигналу (440, 880, 1320, 3000, 6000, 9500, 14000 Гц) охоплюють
    пороги Найквіста для досліджуваних частот 4, 8, 16, 22.05 та 32 кГц, із
    подальшою нормалізацією амплітуди до $[-1.0, 1.0]$.

    Args:
        sample_rate: Частота дискретизації сигналу (Гц).
        duration: Загальна тривалість сигналу у секундах.
        f0: Базова частота основного тону (Гц).

    Returns:
        np.ndarray: Одновимірний масив відліків сигналу (float64).
    """
    total_samples = int(sample_rate * duration)
    t = np.arange(total_samples, dtype=np.float64) / float(sample_rate)

    signal = (
        1.00 * np.sin(2.0 * np.pi * f0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 3000.0 * t)
        + 0.16 * np.sin(2.0 * np.pi * 6000.0 * t)
        + 0.12 * np.sin(2.0 * np.pi * 9500.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 14000.0 * t)
    )

    max_val = np.max(np.abs(signal))
    if max_val > 0.0:
        signal = signal / max_val
    return signal.astype(np.float64)


def save_wav(
    path: Path,
    data: np.ndarray,
    sample_rate: int = config.BASE_SAMPLE_RATE,
    subtype: str = "PCM_16",
) -> None:
    """Зберігає числовий масив відліків у форматі WAV.

    Args:
        path: Цільовий шлях файлу.
        data: Масив амплітуд у діапазоні [-1.0, 1.0].
        sample_rate: Частота дискретизації (Гц).
        subtype: Формат підтипу квантування (наприклад, 'PCM_16').
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path_obj), data, sample_rate, subtype=subtype)


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Зчитує відліки аудіосигналу та частоту дискретизації з WAV-файлу.

    Args:
        path: Шлях до WAV-файлу.

    Returns:
        Tuple[np.ndarray, int]: Кортеж із масиву відліків (float64) та частоти.
    """
    data, sample_rate = sf.read(str(path))
    return data.astype(np.float64), int(sample_rate)


def get_wav_metadata(path: Path) -> Dict[str, Any]:
    """Визначає технічні параметри аудіофайлу за заголовком RIFF/WAV.

    Args:
        path: Шлях до файлу.

    Returns:
        Dict[str, Any]: Словник із параметрами (частота, канали, біти, тривалість).
    """
    info = sf.info(str(path))
    bit_map = {"PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}
    return {
        "file_name": Path(path).name,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "bit_depth": bit_map.get(info.subtype, 16),
        "duration_sec": round(float(info.duration), 3),
        "num_frames": int(info.frames),
        "file_size_bytes": Path(path).stat().st_size,
    }
