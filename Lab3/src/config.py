"""Конфігураційні параметри та константи для Лабораторної роботи №3."""

from pathlib import Path

# Базові шляхи
BASE_DIR: Path = Path(__file__).resolve().parent.parent
AUDIO_DIR: Path = BASE_DIR / "audio"
RESULTS_DIR: Path = BASE_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

AUDIO_FILE_PATH: Path = AUDIO_DIR / "test_signal.wav"

# Параметри синтезу тестового аудіосигналу
SAMPLE_RATE: int = 22050          # Частота дискретизації, Гц
DURATION_SEC: float = 4.0         # Тривалість сигналу, секунди
NOTE_1_FREQ: float = 440.0        # Нота (перші 2 с), Гц
NOTE_2_FREQ: float = 1500.0       # Високий тон (наступні 2 с), Гц
CHIRP_F0: float = 300.0           # Початкова частота лінійного чирпу, Гц
CHIRP_F1: float = 1000.0          # Кінцева частота лінійного чирпу, Гц
CHIRP_AMPLITUDE: float = 0.35     # Відносна амплітуда чирпу

# Базові параметри віконного перетворення STFT
DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_WINDOW: str = "hann"
DEFAULT_N_MELS: int = 128
MAX_FREQ_DISPLAY: float = 2500.0  # Верхня межа частотної осі для чіткості візуалізації

# Сітки параметрів для дослідницького завдання
RESEARCH_N_FFT: list[int] = [256, 1024, 4096]
RESEARCH_HOP_LENGTHS: list[int] = [128, 512, 2048]
RESEARCH_WINDOWS: list[str] = ["boxcar", "hann", "hamming", "blackman"]


def ensure_directories() -> None:
    """Створює всі необхідні каталоги проєкту."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
