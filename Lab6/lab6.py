"""Головний модуль запуску лабораторної роботи № 6.

Тема: Дискретизація та квантування звукових сигналів.
Послідовно виконує всі етапи АЦП, розраховує SNR та зберігає графіки й таблиці.
"""

import csv
import json
import math
import sys

from src import config, experiments, plotting, signal_io


def main() -> int:
    """Головна точка входу лабораторної роботи № 6."""
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА № 6: ОБРОБКА МУЛЬТИМЕДІЙНИХ ДАНИХ")
    print("ТЕМА: ДИСКРЕТИЗАЦІЯ ТА КВАНТУВАННЯ ЗВУКОВИХ СИГНАЛІВ")
    print("=" * 80)

    # 1. Генерація та збереження WAV-файлу
    print("\n--- 1. Створення еталонного аудіосигналу ---")
    raw_signal = signal_io.create_harmonic_signal(
        sample_rate=config.BASE_SAMPLE_RATE,
        duration=config.DURATION_SEC,
        f0=config.BASE_FREQ,
    )
    signal_io.save_wav(
        config.SOURCE_WAV,
        raw_signal,
        sample_rate=config.BASE_SAMPLE_RATE,
        subtype="PCM_16",
    )
    print(f"✔ Сигнал синтезовано та збережено: {config.SOURCE_WAV}")

    # 2. Визначення параметрів WAV-файлу
    print("\n--- 2. Визначення характеристик WAV-файлу ---")
    meta = signal_io.get_wav_metadata(config.SOURCE_WAV)
    ch_label = "Моно" if meta["channels"] == 1 else f"{meta['channels']} канали"
    print(f"  Файл:                   {meta['file_name']}")
    print(f"  Частота дискретизації:  {meta['sample_rate']} Гц")
    print(f"  Кількість каналів:      {meta['channels']} ({ch_label})")
    print(f"  Формат квантування:     {meta['subtype']} ({meta['bit_depth']} біт)")
    print(f"  Тривалість:             {meta['duration_sec']} с")
    print(f"  Кількість відліків:     {meta['num_frames']}")
    print(f"  Розмір на диску:        {meta['file_size_bytes']} байт")

    x, sr = signal_io.load_wav(config.SOURCE_WAV)

    # Побудова графіка вихідної форми сигналу
    wf_path = config.FIGURES_DIR / "original_waveform.png"
    plotting.plot_original_waveform(
        x, sr, wf_path, fragment_ms=config.FRAGMENT_MS
    )
    print(f"✔ Осцилограму збережено: {wf_path}")

    # 3. Повторне дискретизування (Resampling)
    print("\n--- 3. Повторне дискретизування (Resampling) ---")
    resampled_data = experiments.run_resampling_analysis(
        x, sr, config.RESAMPLE_RATES
    )
    resamp_plot_path = config.FIGURES_DIR / "resampling_comparison.png"
    plotting.plot_resampling_comparison(
        resampled_data, resamp_plot_path, fragment_ms=config.FRAGMENT_MS
    )
    print(f"✔ Порівняльний графік ресемплінгу збережено: {resamp_plot_path}")

    # 4. Амплітудне квантування (Quantization) на неквантованому еталоні float64
    print("\n--- 4. Квантування з різною розрядністю ---")
    quantized_data = experiments.run_quantization_analysis(
        raw_signal, config.QUANT_BIT_DEPTHS
    )
    quant_plot_path = config.FIGURES_DIR / "quantization_comparison.png"
    plotting.plot_quantization_comparison(
        raw_signal, sr, quantized_data, quant_plot_path, fragment_ms=8.0
    )
    print(f"✔ Порівняльний графік квантування збережено: {quant_plot_path}")

    # 5. Дослідницька частина: SNR та демонстрація аліасингу
    print("\n--- 5. Дослідження SNR = f(b), SNR = f(fs) та аліасингу ---")
    bits_list, snr_bits, snr_theory = experiments.run_snr_bit_depth_curve(
        raw_signal, config.ALL_BIT_DEPTHS
    )
    rates_list, snr_rates = experiments.run_snr_sample_rate_curve(
        x, sr, config.ALL_RATES
    )
    aliasing_res = experiments.run_aliasing_demo()
    print(
        f"  Аліасинг (1000+5000 Гц -> {aliasing_res['target_sr']} Гц): "
        f"без фільтра пік на {aliasing_res['measured_alias_freq']} Гц ({aliasing_res['naive_alias_db']:.1f} дБ) | "
        f"поліфазний фільтр ({aliasing_res['filtered_alias_db']:.1f} дБ, придушення {aliasing_res['suppression_db']:.1f} дБ)"
    )

    plot_bits_path = config.FIGURES_DIR / "snr_vs_bit_depth.png"
    plotting.plot_snr_vs_bit_depth(
        bits_list, snr_bits, snr_theory, plot_bits_path
    )
    print(f"✔ Графік SNR = f(b) збережено: {plot_bits_path}")

    plot_rates_path = config.FIGURES_DIR / "snr_vs_sample_rate.png"
    plotting.plot_snr_vs_sample_rate(rates_list, snr_rates, plot_rates_path)
    print(f"✔ Графік SNR = f(fs) збережено: {plot_rates_path}")

    alias_plot_path = config.FIGURES_DIR / "aliasing_demonstration.png"
    plotting.plot_aliasing_demonstration(aliasing_res, alias_plot_path)
    print(f"✔ Графік демонстрації аліасингу збережено: {alias_plot_path}")

    # Збереження текстової таблиці результатів
    lines = [
        "ТАБЛИЦЯ 1: Залежність SNR від бітової глибини квантування:",
        f"{'Біти':>5} | {'Рівні сітки':>12} | {'Практичний SNR (дБ)':>20} | {'Теоретичний SNR (дБ)':>21}",
        "-" * 65,
    ]
    csv_rows = []
    for b, sp, st in zip(bits_list, snr_bits, snr_theory):
        levels = 2**b
        lines.append(f"{b:5d} | {levels:12d} | {sp:20.2f} | {st:21.2f}")
        csv_rows.append(
            {
                "Type": "BitDepth",
                "Parameter": b,
                "Levels": levels,
                "SNR_Practical_dB": sp,
                "SNR_Theoretical_dB": st,
            }
        )

    lines.append("\n" + "=" * 65 + "\n")
    lines.append("ТАБЛИЦЯ 2: Залежність SNR від частоти дискретизації (Round-trip):")
    lines.append(f"{'Частота (Гц)':>14} | {'Відновлений SNR (дБ)':>22}")
    lines.append("-" * 40)
    for r, sr_val in zip(rates_list, snr_rates):
        lines.append(f"{r:14d} | {sr_val:22.2f}")
        csv_rows.append(
            {
                "Type": "SampleRate",
                "Parameter": r,
                "Levels": "",
                "SNR_Practical_dB": sr_val,
                "SNR_Theoretical_dB": "",
            }
        )

    table_content = "\n".join(lines)
    txt_table_path = config.TABLES_DIR / "snr_table.txt"
    with open(txt_table_path, "w", encoding="utf-8") as f:
        f.write(table_content)
    print(f"✔ Текстову таблицю збережено: {txt_table_path}")

    # Збереження CSV таблиці
    csv_path = config.TABLES_DIR / "snr_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"✔ CSV таблицю збережено: {csv_path}")

    # Збереження зведеного JSON (суворий RFC 8259 без Infinity)
    summary = {
        "metadata": meta,
        "resample_rates": config.RESAMPLE_RATES,
        "quant_bit_depths": config.QUANT_BIT_DEPTHS,
        "snr_bit_depth": [
            {"bits": b, "snr_practical": sp, "snr_theory": st}
            for b, sp, st in zip(bits_list, snr_bits, snr_theory)
        ],
        "snr_sample_rate": [
            {"rate": r, "snr": (s if math.isfinite(s) else None)}
            for r, s in zip(rates_list, snr_rates)
        ],
        "aliasing_demo": {
            k: v
            for k, v in aliasing_res.items()
            if not k.startswith(("f_", "db_"))
        },
    }
    json_path = config.TABLES_DIR / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"✔ Зведений JSON збережено: {json_path}")

    print("\n" + table_content)
    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНУ РОБОТУ № 6 УСПІШНО ВИКОНАНО!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
