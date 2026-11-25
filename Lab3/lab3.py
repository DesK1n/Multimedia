"""Головний модуль запуску повного конвеєра Лабораторної роботи №3.

Дисципліна: "Обробка мультимедійних даних"
Тема: "Часово-частотний аналіз звукових сигналів"

Скрипт послідовно виконує:
1. Завантаження або синтез тестового аудіосигналу (4 с, 22050 Гц).
2. Побудову часової форми (waveform) та розрахунок фізичних характеристик.
3. Обчислення швидкого перетворення Фур'є (FFT) та побудову амплітудного спектра.
4. Віконне перетворення STFT та побудову спектрограми у децибелах.
5. Обчислення та побудову мел-спектрограми (Mel-Spectrogram).
6. Комплексне порівняння чотирьох представлень сигналу на єдиній панелі.
7. Дослідницьке завдання: аналіз впливу n_fft, hop_length та типу вікна на роздільну здатність.
"""

import csv
import sys
from pathlib import Path

# Додаємо поточну директорію у sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np

from src.config import (
    AUDIO_FILE_PATH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_WINDOW,
    FIGURES_DIR,
    MAX_FREQ_DISPLAY,
    RESEARCH_HOP_LENGTHS,
    RESEARCH_N_FFT,
    RESEARCH_WINDOWS,
    TABLES_DIR,
    ensure_directories,
)
from src.experiments import (
    experiment_hop_length,
    experiment_n_fft,
    experiment_windows,
)
from src.plotting import (
    plot_amplitude_spectrum,
    plot_comparison_panel,
    plot_mel_spectrogram,
    plot_research_hop_length,
    plot_research_n_fft,
    plot_research_window,
    plot_spectrogram,
    plot_waveform,
)
from src.signal_gen import ensure_test_audio_exists
from src.spectral_analysis import (
    compute_fft,
    compute_mel_spectrogram,
    compute_stft,
    mel_to_db,
    stft_to_db,
)


def run_pipeline_lab3() -> None:
    """Виконує повний експериментальний конвеєр Лабораторної роботи №3."""
    print("=" * 75)
    print("ЛАБОРАТОРНА РОБОТА №3: ЧАСОВО-ЧАСТОТНИЙ АНАЛІЗ ЗВУКОВИХ СИГНАЛІВ")
    print("=" * 75)

    ensure_directories()

    # 1. Завантаження або синтез тестового сигналу
    print("\n[Крок 1] Підготовка звукового сигналу...")
    y, sr = ensure_test_audio_exists(AUDIO_FILE_PATH)
    duration = len(y) / sr
    rms_power = float(np.sqrt(np.mean(y ** 2)))
    peak_amp = float(np.max(np.abs(y)))

    print(f"  * Аудіофайл: {AUDIO_FILE_PATH}")
    print(f"  * Частота дискретизації: {sr} Гц")
    print(f"  * Кількість відліків: {len(y)}")
    print(f"  * Тривалість: {duration:.2f} с")
    print(f"  * Пікова амплітуда: {peak_amp:.3f}, RMS енергія: {rms_power:.3f}")

    # Експорт характеристик сигналу у таблицю
    prop_csv = TABLES_DIR / "01_audio_properties.csv"
    with open(prop_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Параметр", "Значення", "Одиниця виміру"])
        writer.writerow(["Частота дискретизації (Sample Rate)", sr, "Гц"])
        writer.writerow(["Тривалість звучання", f"{duration:.3f}", "с"])
        writer.writerow(["Загальна кількість відліків", len(y), "відліків"])
        writer.writerow(["Пікова амплітуда", f"{peak_amp:.4f}", "відн. одн."])
        writer.writerow(["Середньоквадратична амплітуда (RMS)", f"{rms_power:.4f}", "відн. одн."])

    # 2. Часова форма сигналу (Waveform)
    print("\n[Крок 2] Побудова часової форми сигналу (Waveform)...")
    waveform_path = FIGURES_DIR / "01_waveform.png"
    plot_waveform(y, sr, waveform_path)

    # 3. FFT та амплітудний спектр
    print("\n[Крок 3] Обчислення швидкого перетворення Фур'є (FFT)...")
    freqs, magnitude = compute_fft(y, sr)
    fft_path = FIGURES_DIR / "02_fft_spectrum.png"
    plot_amplitude_spectrum(freqs, magnitude, fft_path, max_freq=MAX_FREQ_DISPLAY)

    # 4. STFT та спектрограма
    print("\n[Крок 4] Короткочасне перетворення Фур'є (STFT) та спектрограма...")
    stft_matrix = compute_stft(
        y, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH, window=DEFAULT_WINDOW
    )
    stft_db = stft_to_db(stft_matrix)
    spectrogram_path = FIGURES_DIR / "03_spectrogram.png"
    plot_spectrogram(
        stft_db,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        save_path=spectrogram_path,
        title=f"Спектрограма STFT (n_fft={DEFAULT_N_FFT}, hop={DEFAULT_HOP_LENGTH}, вікно={DEFAULT_WINDOW})",
        max_freq=MAX_FREQ_DISPLAY,
    )
    print(f"  * Розмірність матриці STFT: {stft_matrix.shape} (частотні біни x часові кадри)")

    # 5. Мел-спектрограма
    print("\n[Крок 5] Побудова психоакустичної мел-спектрограми...")
    mel_spec = compute_mel_spectrogram(
        y,
        sr=sr,
        n_fft=DEFAULT_N_FFT,
        hop_length=DEFAULT_HOP_LENGTH,
        n_mels=DEFAULT_N_MELS,
    )
    mel_db = mel_to_db(mel_spec)
    mel_path = FIGURES_DIR / "04_mel_spectrogram.png"
    plot_mel_spectrogram(
        mel_db,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        save_path=mel_path,
        title=f"Мел-спектрограма (n_mels={DEFAULT_N_MELS})",
    )
    print(f"  * Розмірність мел-спектрограми: {mel_spec.shape}")

    # 6. Зведене порівняння чотирьох представлень
    print("\n[Крок 6] Зведена панель порівняння представлень сигналу...")
    comparison_path = FIGURES_DIR / "05_comparison.png"
    plot_comparison_panel(
        y,
        sr=sr,
        freqs=freqs,
        magnitude=magnitude,
        stft_db=stft_db,
        mel_db=mel_db,
        hop_length=DEFAULT_HOP_LENGTH,
        save_path=comparison_path,
        max_freq=MAX_FREQ_DISPLAY,
    )

    # 7. Дослідницьке завдання: вплив n_fft (hop=128 для дотримання COLA при n_fft=256)
    n_fft_hop = DEFAULT_HOP_LENGTH // 4
    print("\n[Крок 7] Дослідження n_fft in {256, 1024, 4096} (компроміс Гейзенберга)...")
    db_n_fft, rows_n_fft = experiment_n_fft(
        y,
        sr=sr,
        n_fft_list=RESEARCH_N_FFT,
        hop_length=n_fft_hop,
        output_dir=TABLES_DIR,
    )
    plot_research_n_fft(
        db_n_fft,
        sr=sr,
        hop_length=n_fft_hop,
        save_path=FIGURES_DIR / "06_research_n_fft.png",
        max_freq=MAX_FREQ_DISPLAY,
    )
    for r in rows_n_fft:
        print(
            f"  * n_fft={r['n_fft']:4d} -> delta_f={r['delta_f_hz']:5.1f} Гц | "
            f"delta_t={r['delta_t_ms']:5.1f} мс | бінів={r['n_bins']} | кадрів={r['n_frames']}"
        )

    # 8. Дослідницьке завдання: вплив hop_length (n_fft=2048 -> перекриття 93.8%, 75.0%, 0.0%)
    print("\n[Крок 8] Дослідження hop_length in {128, 512, 2048}...")
    db_hops, rows_hops = experiment_hop_length(
        y,
        sr=sr,
        hop_list=RESEARCH_HOP_LENGTHS,
        n_fft=DEFAULT_N_FFT,
        output_dir=TABLES_DIR,
    )
    plot_research_hop_length(
        db_hops,
        sr=sr,
        n_fft=DEFAULT_N_FFT,
        save_path=FIGURES_DIR / "07_research_hop_length.png",
        max_freq=MAX_FREQ_DISPLAY,
    )
    for r in rows_hops:
        print(
            f"  * hop={r['hop_length']:4d} -> крок={r['delta_t_hop_ms']:5.1f} мс | "
            f"overlap={r['overlap_pct']:.1f}% | кадрів={r['n_frames']}"
        )

    # 9. Дослідницьке завдання: вплив віконної функції
    print("\n[Крок 9] Дослідження вагових вікон (boxcar, hann, hamming, blackman)...")
    db_wins, rows_wins = experiment_windows(
        y,
        sr=sr,
        windows=RESEARCH_WINDOWS,
        n_fft=DEFAULT_N_FFT // 2,
        hop_length=DEFAULT_HOP_LENGTH // 2,
        output_dir=TABLES_DIR,
    )
    plot_research_window(
        db_wins,
        sr=sr,
        n_fft=DEFAULT_N_FFT // 2,
        hop_length=DEFAULT_HOP_LENGTH // 2,
        save_path=FIGURES_DIR / "08_research_window.png",
        max_freq=MAX_FREQ_DISPLAY,
    )
    for r in rows_wins:
        print(
            f"  * Вікно '{r['window']:8s}' -> головна пелюстка={r['main_lobe']} | "
            f"придушення бічних={r['side_lobe']}"
        )

    print("\n" + "=" * 75)
    print(f"[УСПІХ] Лабораторна робота №3 виконана! Графіки: {FIGURES_DIR}, Таблиці: {TABLES_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline_lab3()
