"""Модуль проведення дослідницьких експериментів впливу параметрів STFT."""

import csv
from pathlib import Path

import numpy as np

from src.spectral_analysis import compute_stft, stft_to_db


def experiment_n_fft(
    y: np.ndarray,
    sr: int,
    n_fft_list: list[int],
    hop_length: int,
    output_dir: Path,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    r"""Досліджує компроміс частотної та часової роздільної здатності Гейзенберга-Габора.

    Співвідношення невизначеності:
    $$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}, \quad \Delta f = \frac{f_s}{N_{\text{fft}}}, \quad \Delta t_{\text{window}} = \frac{N_{\text{fft}}}{f_s}$$

    Args:
        y (np.ndarray): Вхідний аудіосигнал.
        sr (int): Частота дискретизації, Гц.
        n_fft_list (list[int]): Список розмірів вікон аналізу (наприклад, [256, 1024, 4096]).
        hop_length (int): Фіксований крок зміщення кадру.
        output_dir (Path): Каталог для збереження таблиці результатів.

    Returns:
        tuple[dict[int, np.ndarray], list[dict]]: Словник обчислених dB-спектрограм та список числових метрик.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dict: dict[int, np.ndarray] = {}
    rows: list[dict] = []

    for n_fft in n_fft_list:
        stft_mat = compute_stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
        db_mat = stft_to_db(stft_mat)
        db_dict[n_fft] = db_mat

        delta_f = sr / n_fft
        delta_t_ms = (n_fft / sr) * 1000.0
        n_bins = 1 + n_fft // 2
        n_frames = db_mat.shape[1]

        rows.append({
            "n_fft": n_fft,
            "delta_f_hz": delta_f,
            "delta_t_ms": delta_t_ms,
            "n_bins": n_bins,
            "n_frames": n_frames,
        })

    csv_path = output_dir / "06_research_n_fft.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Розмір n_fft",
            "Частотна роздільність delta_f, Гц",
            "Часове вікно delta_t, мс",
            "Частотних бінів (N/2 + 1)",
            "Кадрів у часі",
        ])
        for r in rows:
            writer.writerow([
                r["n_fft"],
                f"{r['delta_f_hz']:.2f}",
                f"{r['delta_t_ms']:.2f}",
                r["n_bins"],
                r["n_frames"],
            ])

    print(f"[INFO] Результати дослідження n_fft збережено: {csv_path}")
    return db_dict, rows


def experiment_hop_length(
    y: np.ndarray,
    sr: int,
    hop_list: list[int],
    n_fft: int,
    output_dir: Path,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    r"""Досліджує крок дискретизації за часом (hop length) та ступінь перекриття кадрів.

    Часовий крок між сусідніми стовпцями:
    $$\Delta t_{\text{hop}} = \frac{H}{f_s}, \quad \text{Overlap} = \frac{N_{\text{fft}} - H}{N_{\text{fft}}} \cdot 100\%$$

    Args:
        y (np.ndarray): Вхідний аудіосигнал.
        sr (int): Частота дискретизації.
        hop_list (list[int]): Список значень hop_length (наприклад, [128, 512, 2048]).
        n_fft (int): Фіксований розмір вікна FFT.
        output_dir (Path): Каталог для експорту CSV.

    Returns:
        tuple[dict[int, np.ndarray], list[dict]]: Словник спектрограм та числові показники.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dict: dict[int, np.ndarray] = {}
    rows: list[dict] = []

    for hop in hop_list:
        stft_mat = compute_stft(y, n_fft=n_fft, hop_length=hop, window="hann")
        db_mat = stft_to_db(stft_mat)
        db_dict[hop] = db_mat

        delta_t_hop_ms = (hop / sr) * 1000.0
        overlap_pct = (n_fft - hop) / n_fft * 100.0
        n_frames = db_mat.shape[1]

        rows.append({
            "hop_length": hop,
            "delta_t_hop_ms": delta_t_hop_ms,
            "overlap_pct": overlap_pct,
            "n_frames": n_frames,
        })

    csv_path = output_dir / "07_research_hop_length.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Параметр hop_length",
            "Крок у часі delta_t_hop, мс",
            "Перекриття вікон (Overlap), %",
            "Загальна кількість кадрів",
        ])
        for r in rows:
            writer.writerow([
                r["hop_length"],
                f"{r['delta_t_hop_ms']:.2f}",
                f"{r['overlap_pct']:.1f}%",
                r["n_frames"],
            ])

    print(f"[INFO] Результати дослідження hop_length збережено: {csv_path}")
    return db_dict, rows


def experiment_windows(
    y: np.ndarray,
    sr: int,
    windows: list[str],
    n_fft: int,
    hop_length: int,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Досліджує вплив форми вагового вікна на пригнічення ефекту витоку спектра (Spectral Leakage).

    Args:
        y (np.ndarray): Вхідний аудіосигнал.
        sr (int): Частота дискретизації.
        windows (list[str]): Список віконних функцій ("boxcar", "hann", "hamming", "blackman").
        n_fft (int): Розмір вікна FFT.
        hop_length (int): Крок зміщення вікна.
        output_dir (Path): Каталог для експорту CSV.

    Returns:
        tuple[dict[str, np.ndarray], list[dict]]: Словник спектрограм та аналітичний опис вікон.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dict: dict[str, np.ndarray] = {}

    properties = {
        "boxcar": {
            "main_lobe_bin": "2 bin",
            "side_lobe_suppr_db": "-13 дБ",
            "leakage": "Максимальний (різкі розриви на краях кадру)",
            "use_case": "Аналіз перехідних імпульсів або періодичних сигналів, узгоджених з сіткою",
        },
        "hann": {
            "main_lobe_bin": "4 bin",
            "side_lobe_suppr_db": "-31 дБ",
            "leakage": "Низький (гладкий спад до нуля на краях)",
            "use_case": "Універсальний стандарт для аудіоаналізу та мовленнєвих задач",
        },
        "hamming": {
            "main_lobe_bin": "4 bin",
            "side_lobe_suppr_db": "-43 дБ (1-й пел.)",
            "leakage": "Помірний дальній фон (-6 дБ/окт через w[0]=0.08)",
            "use_case": "Вузькосмуговий аналіз поблизу носія; гірша ізоляція далеких частот",
        },
        "blackman": {
            "main_lobe_bin": "6 bin",
            "side_lobe_suppr_db": "-58 дБ",
            "leakage": "Мінімальний (глибоке придушення бічних пелюстків)",
            "use_case": "Виявлення слабких гармонік на фоні потужних сусідніх частот",
        },
    }

    rows: list[dict] = []
    for win in windows:
        stft_mat = compute_stft(y, n_fft=n_fft, hop_length=hop_length, window=win)
        db_mat = stft_to_db(stft_mat)
        db_dict[win] = db_mat

        # Емпіричне вимірювання позасмугової шумової полиці (f > 2000 Гц, t in [2.3, 3.7] с)
        freqs = np.linspace(0.0, sr / 2.0, db_mat.shape[0])
        times = np.arange(db_mat.shape[1]) * hop_length / float(sr)
        f_mask = freqs > 2000.0
        t_mask = (times >= 2.3) & (times <= 3.7)
        if np.any(f_mask) and np.any(t_mask):
            region = db_mat[np.ix_(f_mask, t_mask)]
            noise_floor_med = float(np.median(region))
            noise_floor_mean = float(np.mean(region))
        else:
            noise_floor_med = float(np.median(db_mat))
            noise_floor_mean = float(np.mean(db_mat))

        prop = properties.get(win, {})
        rows.append({
            "window": win,
            "main_lobe": prop.get("main_lobe_bin", "N/A"),
            "side_lobe": prop.get("side_lobe_suppr_db", "N/A"),
            "noise_floor_med_db": noise_floor_med,
            "noise_floor_mean_db": noise_floor_mean,
            "leakage": prop.get("leakage", "N/A"),
            "use_case": prop.get("use_case", "N/A"),
        })

    csv_path = output_dir / "08_research_window.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Віконна функція",
            "Ширина головної пелюстки",
            "Рівень 1-го бічного пелюстка",
            "Виміряний фон витоку (>2 кГц, медіана), дБ",
            "Виміряний фон витоку (>2 кГц, середнє), дБ",
            "Характер спектрального витоку (Leakage)",
            "Рекомендована сфера застосування",
        ])
        for r in rows:
            writer.writerow([
                r["window"],
                r["main_lobe"],
                r["side_lobe"],
                f"{r['noise_floor_med_db']:.2f}",
                f"{r['noise_floor_mean_db']:.2f}",
                r["leakage"],
                r["use_case"],
            ])

    print(f"[INFO] Результати дослідження вікон збережено: {csv_path}")
    return db_dict, rows
