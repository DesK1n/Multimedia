"""Модуль візуалізації результатів дискретизації та квантування.

Будує осцилограми сигналів, порівняльні часові діаграми ресемплінгу та
квантування, а також графіки залежностей SNR від параметрів оцифрування.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_original_waveform(
    x: np.ndarray,
    sr: int,
    save_path: Path,
    fragment_ms: float = 10.0,
) -> None:
    """Будує осцилограму часового фрагмента вихідного сигналу.

    Args:
        x: Відліки сигналу.
        sr: Частота дискретизації (Гц).
        save_path: Шлях для збереження зображення.
        fragment_ms: Тривалість відображуваного фрагмента (мс).
    """
    n_samples = int(sr * fragment_ms / 1000.0)
    t_ms = np.arange(n_samples) / float(sr) * 1000.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        t_ms,
        x[:n_samples],
        marker=".",
        markersize=4,
        color="#1f77b4",
        linewidth=1.5,
        label="Відліки сигналу",
    )
    ax.set_title(
        f"Фрагмент вихідного сигналу ({sr} Гц, 16-біт PCM)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Час (мілісекунди, мс)", fontsize=10)
    ax.set_ylabel("Нормована амплітуда", fontsize=10)
    ax.set_ylim(-1.15, 1.38)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_resampling_comparison(
    resampled_dict: Dict[int, np.ndarray],
    save_path: Path,
    fragment_ms: float = 10.0,
) -> None:
    """Будує порівняльні осцилограми сигналу при різних частотах дискретизації.

    Args:
        resampled_dict: Словник {частота: масив_відліків}.
        save_path: Шлях збереження графіка.
        fragment_ms: Тривалість фрагмента (мс).
    """
    rates = sorted(resampled_dict.keys())
    fig, axes = plt.subplots(len(rates), 1, figsize=(9, 8.5), sharex=True)
    fig.suptitle(
        "Порівняння форми сигналу при різних частотах дискретизації (Resampling)",
        fontsize=12,
        fontweight="bold",
    )

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for ax, sr, color in zip(axes, rates, colors):
        y = resampled_dict[sr]
        n_samples = int(sr * fragment_ms / 1000.0)
        t_ms = np.arange(n_samples) / float(sr) * 1000.0

        ax.plot(
            t_ms,
            y[:n_samples],
            marker="o",
            markersize=3,
            color=color,
            linewidth=1.2,
            label=f"{sr} Гц ({n_samples} відліків на {fragment_ms} мс)",
        )
        ax.set_ylabel(f"{sr} Гц", fontsize=10, fontweight="bold")
        ax.set_xlim(0.0, fragment_ms)
        ax.set_ylim(-1.15, 1.40)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Час (мілісекунди, мс)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _levels_word(levels: int) -> str:
    """Повертає граматично правильне слово 'рівні'/'рівнів' для української мови."""
    if 11 <= levels % 100 <= 14:
        return "рівнів"
    if levels % 10 in (2, 3, 4):
        return "рівні"
    return "рівнів"


def plot_quantization_comparison(
    x: np.ndarray,
    sr: int,
    quantized_dict: Dict[int, np.ndarray],
    save_path: Path,
    fragment_ms: float = 8.0,
) -> None:
    """Будує графіки квантованого сигналу при різній бітовій глибині.

    Args:
        x: Вихідний еталонний сигнал.
        sr: Базова частота дискретизації.
        quantized_dict: Словник {розрядність: квантований_сигнал}.
        save_path: Шлях для збереження зображення.
        fragment_ms: Тривалість відображуваного фрагмента (мс).
    """
    depths = sorted(quantized_dict.keys())
    n_samples = int(sr * fragment_ms / 1000.0)
    t_ms = np.arange(n_samples) / float(sr) * 1000.0

    fig, axes = plt.subplots(len(depths), 1, figsize=(9, 9), sharex=True)
    fig.suptitle(
        "Спотворення форми сигналу при різній бітовій глибині (Quantization)",
        fontsize=12,
        fontweight="bold",
    )

    colors = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]

    for ax, b, color in zip(axes, depths, colors):
        y = quantized_dict[b]
        levels = 2**b
        lvl_word = _levels_word(levels)

        ax.plot(
            t_ms,
            x[:n_samples],
            label="Оригінальний неперервний сигнал",
            color="gray",
            linewidth=1.1,
            alpha=0.65,
        )
        if b <= 4:
            ax.step(
                t_ms,
                y[:n_samples],
                where="mid",
                label=f"{b}-бітне квантування ({levels} {lvl_word})",
                color=color,
                linewidth=1.5,
            )
        else:
            ax.plot(
                t_ms,
                y[:n_samples],
                label=f"{b}-бітне квантування ({levels} {lvl_word})",
                color=color,
                linewidth=1.4,
            )
        ax.set_ylabel(f"{b} біт\n({levels} {lvl_word})", fontsize=9, fontweight="bold")
        ax.set_ylim(-1.15, 1.42)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Час (мілісекунди, мс)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snr_vs_bit_depth(
    bits: List[int],
    snr_practical: List[float],
    snr_theory: List[float],
    save_path: Path,
) -> None:
    """Будує графік залежності SNR від бітової глибини квантування.

    Args:
        bits: Список розрядностей.
        snr_practical: Експериментальні значення SNR (дБ).
        snr_theory: Теоретичні значення 6.02b + 1.76 (дБ).
        save_path: Шлях для збереження зображення.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        bits,
        snr_practical,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Практичний SNR (експеримент)",
    )
    ax.plot(
        bits,
        snr_theory,
        linestyle="--",
        color="#d62728",
        linewidth=2.0,
        label=r"Теоретична межа ($6.02 \cdot b + 1.76$ дБ)",
    )

    ax.set_title(
        "Залежність відношення сигнал/шум (SNR) від бітової глибини",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Бітова глибина (розрядність, b біт)", fontsize=11)
    ax.set_ylabel("Відношення сигнал/шум (SNR, дБ)", fontsize=11)
    ax.set_xticks(bits)
    ax.set_ylim(0, max(max(snr_practical), max(snr_theory)) + 14.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")

    # Підписи значень нижче кривої, щоб не перетинатися з червоною теоретичною лінією
    for b, s in zip(bits, snr_practical):
        if b in [2, 4, 8, 12, 16]:
            ax.annotate(
                f"{s:.1f} дБ",
                (b, s),
                textcoords="offset points",
                xytext=(8, -14),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1f77b4",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snr_vs_sample_rate(
    rates: List[int],
    snr_rates: List[float],
    save_path: Path,
    harmonic_freqs: Optional[List[float]] = None,
) -> None:
    """Будує наочний графік залежності SNR від частоти дискретизації та межі Найквіста.

    Args:
        rates: Список досліджуваних частот дискретизації (Гц).
        snr_rates: Значення відновленого SNR після round-trip ресемплінгу (дБ).
        save_path: Шлях для збереження графіка.
        harmonic_freqs: Перелік частот гармонік сигналу (Гц) для підрахунку втрат.
    """
    if harmonic_freqs is None:
        harmonic_freqs = [440.0, 880.0, 1320.0, 3000.0, 6000.0, 9500.0, 14000.0]

    f_max = max(harmonic_freqs)
    nyquist_fs = 2.0 * f_max
    total_harm = len(harmonic_freqs)

    finite_pairs = [(r, s) for r, s in zip(rates, snr_rates) if np.isfinite(s)]
    inf_rates = [r for r, s in zip(rates, snr_rates) if not np.isfinite(s)]

    inf_display_y = 51.0
    x_min = min(rates) - 2000
    x_max = max(rates) + 3200

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.axvspan(x_min, nyquist_fs, color="#ffebee", alpha=0.55, zorder=0)
    ax.axvspan(nyquist_fs, x_max, color="#e8f5e9", alpha=0.55, zorder=0)
    ax.axvline(
        nyquist_fs,
        color="#d32f2f",
        linestyle="--",
        linewidth=1.8,
        zorder=2,
        label=f"Поріг Найквіста 2·f_max = {nyquist_fs:.0f} Гц",
    )

    ax.text(
        5200,
        54.5,
        f"Зона втрат (fs < 2·f_max = {nyquist_fs / 1000:.0f} кГц)\n"
        "Фільтр відрізає верхні гармоніки",
        fontsize=9.5,
        fontweight="bold",
        color="#b71c1c",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#ef9a9a"},
    )
    ax.text(
        29200,
        54.5,
        f"Безпечна зона Найквіста\n"
        f"(fs ≥ {nyquist_fs / 1000:.0f} кГц: усі {total_harm} гармонік збережено)",
        fontsize=9.5,
        fontweight="bold",
        color="#1b5e20",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#a5d6a7"},
    )

    f_rates = [r for r, _ in finite_pairs]
    f_snrs = [s for _, s in finite_pairs]
    ax.plot(
        f_rates,
        f_snrs,
        marker="s",
        markersize=7,
        linewidth=2.5,
        color="#e65100",
        zorder=4,
        label="SNR після round-trip ресемплінгу",
    )

    if inf_rates:
        ax.scatter(
            inf_rates,
            [inf_display_y] * len(inf_rates),
            marker="*",
            s=200,
            color="#2ca02c",
            zorder=5,
            label="Еталон 44.1 кГц (SNR = ∞, без втрат)",
        )
        for r in inf_rates:
            ax.annotate(
                f"∞ дБ\n(оригінал {total_harm}/{total_harm})",
                (r, inf_display_y),
                textcoords="offset points",
                xytext=(-6, -28),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1b5e20",
            )

    offsets = {
        4000: (6, 10),
        8000: (-4, -27),
        11025: (8, 10),
        16000: (0, -27),
        22050: (-18, 10),
        32000: (0, 10),
    }
    for r, s in finite_pairs:
        passed = sum(1 for f in harmonic_freqs if f < r / 2.0)
        offset = offsets.get(r, (0, 10))
        ax.annotate(
            f"{s:.1f} дБ\n({passed}/{total_harm} гарм.)",
            (r, s),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#bf360c",
        )

    ax.annotate(
        "Стрибок +23.8 дБ:\nпройшли всі 7 гармонік",
        xy=(27200, 35.0),
        xytext=(15800, 37.0),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#b71c1c",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff8e1", "edgecolor": "#ffb74d"},
        arrowprops={"arrowstyle": "->", "color": "#d32f2f", "lw": 1.6},
        zorder=6,
    )

    ax.set_title(
        "Залежність якості відновлення (SNR) від частоти дискретизації f_s\n"
        "(7 гармонік: 440–14 000 Гц; round-trip 44.1 кГц → f_s → 44.1 кГц)",
        fontsize=11.5,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Частота дискретизації f_s (Гц)", fontsize=10.5)
    ax.set_ylabel("Відношення сигнал/шум (SNR, дБ)", fontsize=10.5)
    ax.set_xticks(rates)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 63)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    secax = ax.secondary_xaxis(
        "top", functions=(lambda fs: fs / 2000.0, lambda fn: fn * 2000.0)
    )
    secax.set_xlabel("Межа Найквіста f_N = f_s / 2 (кГц)", fontsize=10)
    secax.set_xticks([2, 4, 5.5, 8, 11, 14, 16, 22.05])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aliasing_demonstration(
    aliasing_res: Dict[str, Any], save_path: Path
) -> None:
    """Будує 3-панельне спектральне порівняння ефекту аліасингу та антиаліасингу.

    Усі три підграфіки мають спільну вісь частот 0–6500 Гц для наочного
    зіставлення положення корисної гармоніки, межі Найквіста і дзеркального аліасу.

    Args:
        aliasing_res: Результати експерименту run_aliasing_demo().
        save_path: Шлях для збереження графіка.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), sharex=True)
    nyq = float(aliasing_res["nyquist_target"])
    low_f = float(aliasing_res["low_freq"])
    high_f = float(aliasing_res["high_freq"])
    alias_f = float(aliasing_res["expected_alias_freq"])
    naive_db = float(aliasing_res["naive_alias_db"])
    filt_db = float(aliasing_res["filtered_alias_db"])
    supp_db = float(aliasing_res["suppression_db"])
    excess_hz = high_f - nyq

    # 1. Оригінальний спектр (44100 Гц)
    ax0 = axes[0]
    ax0.axvspan(0, nyq, color="#e8f5e9", alpha=0.45, zorder=0)
    ax0.axvspan(nyq, 6500, color="#ffebee", alpha=0.55, zorder=0)
    ax0.plot(
        aliasing_res["f_orig"],
        aliasing_res["db_orig"],
        color="#1f77b4",
        linewidth=2.0,
        zorder=3,
    )
    ax0.axvline(nyq, color="#d62728", linestyle="--", linewidth=1.7, zorder=4)
    ax0.text(
        2250,
        -84,
        "Дозволена зона (< f_N)",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#2e7d32",
    )
    ax0.text(
        5450,
        -84,
        "Заборонена зона (> f_N)",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#c62828",
    )
    ax0.text(
        nyq - 70,
        -70,
        f"Межа Найквіста\nf_N = {nyq:.0f} Гц",
        ha="right",
        fontsize=9,
        fontweight="bold",
        color="#d62728",
    )
    ax0.annotate(
        f"Корисний тон\n{low_f:.0f} Гц (0.0 дБ)",
        xy=(low_f, -1.0),
        xytext=(1450, -22),
        fontsize=9,
        fontweight="bold",
        color="#0d47a1",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#90caf9"},
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.4},
    )
    ax0.annotate(
        f"Високий тон {high_f:.0f} Гц\n(+{excess_hz:.0f} Гц вище f_N!)",
        xy=(high_f + 20, -2.0),
        xytext=(5650, -24),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#b71c1c",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffebee", "edgecolor": "#ef9a9a"},
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.4},
    )
    ax0.annotate(
        "",
        xy=(high_f, -48),
        xytext=(nyq, -48),
        arrowprops={"arrowstyle": "<->", "color": "#b71c1c", "lw": 1.6},
    )
    ax0.text(
        (nyq + high_f) / 2.0,
        -44,
        f"+{excess_hz:.0f} Гц",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#b71c1c",
    )
    ax0.set_title(
        f"1. Вихідний спектр (fs = {aliasing_res['orig_sr']} Гц): "
        f"корисна {low_f:.0f} Гц + високочастотна {high_f:.0f} Гц",
        fontsize=11,
        fontweight="bold",
    )
    ax0.set_ylabel("Магнітуда (дБ)", fontsize=10)
    ax0.set_ylim(-92, 10)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # 2. Наївна децимація без фільтра (8820 Гц)
    ax1 = axes[1]
    ax1.axvspan(
        nyq,
        6500,
        facecolor="#eceff1",
        alpha=0.85,
        hatch="///",
        edgecolor="#b0bec5",
        zorder=0,
    )
    ax1.axvline(nyq, color="#d62728", linestyle="--", linewidth=1.7, zorder=4)
    ax1.axvline(high_f, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax1.text(
        high_f + 60,
        -14,
        f"Було {high_f:.0f} Гц",
        fontsize=8.5,
        color="#546e7a",
        fontweight="bold",
    )
    ax1.text(
        5450,
        -48,
        f"Зона > {nyq:.0f} Гц\nне існує при fs = {aliasing_res['target_sr']} Гц",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#37474f",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#b0bec5"},
    )
    ax1.plot(
        aliasing_res["f_naive"],
        aliasing_res["db_naive"],
        color="#d62728",
        linewidth=2.0,
        zorder=3,
    )
    ax1.annotate(
        "",
        xy=(alias_f + 30, -4),
        xytext=(high_f, -24),
        arrowprops={
            "arrowstyle": "->",
            "color": "#d62728",
            "lw": 2.0,
            "connectionstyle": "arc3,rad=-0.30",
        },
        zorder=5,
    )
    ax1.annotate(
        f"Корисний тон\n{low_f:.0f} Гц (зберігся)",
        xy=(low_f, -1.0),
        xytext=(1450, -22),
        fontsize=9,
        fontweight="bold",
        color="#0d47a1",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#90caf9"},
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.4},
    )
    ax1.annotate(
        f"АЛІАС (дзеркальне відбиття)!\n"
        f"|{high_f:.0f} - {aliasing_res['target_sr']}| = {alias_f:.0f} Гц\n"
        f"Рівень: {naive_db:.1f} дБ (фальшивий звук!)",
        xy=(alias_f - 20, -6),
        xytext=(2450, -50),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#b71c1c",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffebee", "edgecolor": "#d62728"},
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.5},
        zorder=6,
    )
    ax1.set_title(
        f"2. БЕЗ антиаліасингу (x[::5], fs = {aliasing_res['target_sr']} Гц): "
        f"тон {high_f:.0f} Гц відбився у фальшиву частоту {alias_f:.0f} Гц",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Магнітуда (дБ)", fontsize=10)
    ax1.set_ylim(-92, 10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 3. Поліфазний ресемплінг з антиаліасингом (8820 Гц)
    ax2 = axes[2]
    ax2.axvspan(
        nyq,
        6500,
        facecolor="#eceff1",
        alpha=0.85,
        hatch="///",
        edgecolor="#b0bec5",
        zorder=0,
    )
    ax2.axvline(nyq, color="#d62728", linestyle="--", linewidth=1.7, zorder=4)
    ax2.axvline(alias_f, color="gray", linestyle=":", linewidth=1.4, zorder=2)
    ax2.text(
        5450,
        -48,
        f"Зона > {nyq:.0f} Гц\nвідрізана фільтром ДО децимації",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#2e7d32",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#a5d6a7"},
    )
    ax2.plot(
        aliasing_res["f_filt"],
        aliasing_res["db_filt"],
        color="#2ca02c",
        linewidth=2.0,
        zorder=3,
    )
    ax2.annotate(
        f"Корисний тон\n{low_f:.0f} Гц (чистий!)",
        xy=(low_f, -1.0),
        xytext=(1450, -22),
        fontsize=9,
        fontweight="bold",
        color="#1b5e20",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#a5d6a7"},
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.4},
    )
    ax2.annotate(
        f"Аліас на {alias_f:.0f} Гц придушено!\n"
        f"Було {naive_db:.1f} дБ → стало {filt_db:.1f} дБ\n"
        f"(виграш {supp_db:.1f} дБ)",
        xy=(alias_f - 20, filt_db),
        xytext=(2450, -50),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#1b5e20",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#e8f5e9", "edgecolor": "#2ca02c"},
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.5},
        zorder=6,
    )
    ax2.set_title(
        f"3. З антиаліасингом (resample_poly, fs = {aliasing_res['target_sr']} Гц): "
        f"залишився чистий тон {low_f:.0f} Гц",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Частота (Гц)", fontsize=10.5)
    ax2.set_ylabel("Магнітуда (дБ)", fontsize=10)
    ax2.set_xlim(0, 6500)
    ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
    ax2.set_ylim(-92, 10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
