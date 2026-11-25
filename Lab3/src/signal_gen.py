"""Модуль генерації та завантаження тестових звукових сигналів."""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from src.config import (
    AUDIO_FILE_PATH,
    CHIRP_AMPLITUDE,
    CHIRP_F0,
    CHIRP_F1,
    DURATION_SEC,
    NOTE_1_FREQ,
    NOTE_2_FREQ,
    SAMPLE_RATE,
)


def generate_test_signal(
    sample_rate: int = SAMPLE_RATE,
    duration_sec: float = DURATION_SEC,
    note1_freq: float = NOTE_1_FREQ,
    note2_freq: float = NOTE_2_FREQ,
    chirp_f0: float = CHIRP_F0,
    chirp_f1: float = CHIRP_F1,
    chirp_amp: float = CHIRP_AMPLITUDE,
) -> tuple[np.ndarray, int]:
    r"""Генерує еталонний тестовий сигнал, що поєднує стаціонарні гармоніки та нестаціонарний чирп.

    Структура сигналу:
    - Перша половина часу $t \in [0, T/2]$: гармонійне коливання $x_1(t) = \sin(2\pi f_1 t)$;
    - Друга половина часу $t \in [T/2, T]$: гармонійне коливання $x_2(t) = \sin(2\pi f_2 t)$;
    - По всій тривалості накладено лінійний чирп із миттєвою частотою $f(t) = f_0 + \frac{f_1 - f_0}{T} t$:
      $$\phi(t) = 2\pi \left( f_0 t + \frac{f_1 - f_0}{2T} t^2 \right), \quad x_{\text{chirp}}(t) = A_c \cdot \sin(\phi(t))$$

    Args:
        sample_rate (int): Частота дискретизації у Гц.
        duration_sec (float): Тривалість сигналу у секундах.
        note1_freq (float): Частота першого тону (Гц).
        note2_freq (float): Частота другого тону (Гц).
        chirp_f0 (float): Початкова частота чирпу (Гц).
        chirp_f1 (float): Кінцева частота чирпу (Гц).
        chirp_amp (float): Амплітуда чирпу відносно основних тонів.

    Returns:
        tuple[np.ndarray, int]: Масив відліків сигналу (float32, діапазон [-1, 1]) та частота дискретизації.
    """
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    half = n_samples // 2

    # Послідовні гармонічні складові
    tones = np.zeros(n_samples, dtype=np.float64)
    tones[:half] = np.sin(2.0 * np.pi * note1_freq * t[:half])
    tones[half:] = np.sin(2.0 * np.pi * note2_freq * t[half:])

    # Неперервний лінійно-частотно модульований імпульс (чирп)
    chirp_rate = (chirp_f1 - chirp_f0) / duration_sec
    chirp_phase = 2.0 * np.pi * (chirp_f0 * t + 0.5 * chirp_rate * (t ** 2))
    chirp = chirp_amp * np.sin(chirp_phase)

    # Підсумкова суперпозиція хвиль
    raw_signal = tones + chirp

    # Нормалізація амплітуди для запобігання кліпінгу
    max_amp = np.max(np.abs(raw_signal))
    normalized = raw_signal / max_amp if max_amp > 0 else raw_signal

    return normalized.astype(np.float32), sample_rate


def save_audio(path: Path | str, signal: np.ndarray, sr: int) -> None:
    """Зберігає числовий масив відліків у аудіофайл формату WAV."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path_obj), signal, sr, subtype="PCM_16")
    print(f"[INFO] Аудіосигнал успішно збережено: {path_obj}")


def load_audio(
    path: Path | str, sr: Optional[int] = None
) -> tuple[np.ndarray, int]:
    """Завантажує звуковий файл у пам'ять за допомогою librosa (моно-сигнал)."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Аудіофайл не знайдено: {path_obj}")

    y, sample_rate = librosa.load(str(path_obj), sr=sr, mono=True)
    return y.astype(np.float32), int(sample_rate)


def ensure_test_audio_exists(path: Path | str = AUDIO_FILE_PATH) -> tuple[np.ndarray, int]:
    """Перевіряє наявність аудіофайлу на диску і генерує його при відсутності."""
    path_obj = Path(path)
    if path_obj.is_file():
        print(f"[INFO] Використовується наявний аудіофайл: {path_obj}")
        return load_audio(path_obj)

    print(f"[INFO] Файл {path_obj} відсутній -> генерую синтетичний тестовий аудіосигнал...")
    sig, sr = generate_test_signal()
    save_audio(path_obj, sig, sr)
    return sig, sr
