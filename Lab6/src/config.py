"""Конфігурація лабораторної роботи № 6.

Визначає глобальні константи експериментів: шляхи до каталогів,
базову частоту дискретизації, параметри синтезу сигналу, набір частот
ресемплінгу та бітових глибин квантування.
"""

from pathlib import Path
from typing import Final, List

# Базові каталоги проєкту
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
SRC_DIR: Final[Path] = BASE_DIR / "src"
RESULTS_DIR: Final[Path] = BASE_DIR / "results"
AUDIO_DIR: Final[Path] = RESULTS_DIR / "audio"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"

for directory in (AUDIO_DIR, FIGURES_DIR, TABLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Шлях до еталонного WAV-файлу
SOURCE_WAV: Final[Path] = AUDIO_DIR / "tone.wav"

# Базові параметри синтезу сигналу
BASE_SAMPLE_RATE: Final[int] = 44100
DURATION_SEC: Final[float] = 2.0
BASE_FREQ: Final[float] = 440.0  # Нота (A4)

# Частоти для повторного дискретизування (Resampling)
RESAMPLE_RATES: Final[List[int]] = [8000, 16000, 22050, 44100]

# Бітові глибини для детального візуального порівняння квантування
QUANT_BIT_DEPTHS: Final[List[int]] = [2, 4, 8, 16]

# Повний діапазон для побудови кривої SNR = f(b)
ALL_BIT_DEPTHS: Final[List[int]] = list(range(2, 17))

# Діапазон частот для побудови кривої SNR = f(fs)
ALL_RATES: Final[List[int]] = [4000, 8000, 11025, 16000, 22050, 32000, 44100]

# Тривалість фрагмента для відображення осцилограм (мс)
FRAGMENT_MS: Final[float] = 10.0
