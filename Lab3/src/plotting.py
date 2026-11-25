"""Модуль побудови та експорту візуалізацій для Лабораторної роботи №3."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(y: np.ndarray, sr: int, save_path: Path) -> None:
    """Будує часову форму аудіосигналу (Waveform)."""
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, color="#1f77b4", linewidth=0.6)
    ax.set_title("Часова форма звукового сигналу (Waveform)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Час, с", fontsize=11)
    ax.set_ylabel("Амплітуда", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік часової форми збережено: {save_path}")


def plot_amplitude_spectrum(
    freqs: np.ndarray, magnitude: np.ndarray, save_path: Path, max_freq: float = 2500.0
) -> None:
    """Будує амплітудний спектр швидкого перетворення Фур'є (FFT)."""
    mask = freqs <= max_freq
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[mask], magnitude[mask], color="#2ca02c", linewidth=0.8)
    ax.set_title("Амплітудний спектр Фур'є (FFT для всього сигналу)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Частота, Гц", fontsize=11)
    ax.set_ylabel("Амплітуда |X(f)|", fontsize=11)
    ax.set_xlim(0, max_freq)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік амплітудного спектра збережено: {save_path}")


def plot_spectrogram(
    db_matrix: np.ndarray,
    sr: int,
    hop_length: int,
    save_path: Path,
    title: str = "Спектрограма (STFT)",
    max_freq: float = 2500.0,
) -> None:
    """Будує спектрограму короткочасного перетворення Фур'є у шкалі децибелів."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    img = librosa.display.specshow(
        db_matrix,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    ax.set_ylim(0, max_freq)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Час, с", fontsize=11)
    ax.set_ylabel("Частота, Гц", fontsize=11)
    fig.colorbar(img, ax=ax, format="%+2.0f дБ")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Спектрограму збережено: {save_path}")


def plot_mel_spectrogram(
    mel_db: np.ndarray,
    sr: int,
    hop_length: int,
    save_path: Path,
    title: str = "Мел-спектрограма (Mel-Spectrogram)",
) -> None:
    """Будує мел-спектрограму звукового сигналу."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        htk=True,
        ax=ax,
        cmap="viridis",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Час, с", fontsize=11)
    ax.set_ylabel("Частота (Мел-шкала)", fontsize=11)
    fig.colorbar(img, ax=ax, format="%+2.0f дБ")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Мел-спектрограму збережено: {save_path}")


def plot_comparison_panel(
    y: np.ndarray,
    sr: int,
    freqs: np.ndarray,
    magnitude: np.ndarray,
    stft_db: np.ndarray,
    mel_db: np.ndarray,
    hop_length: int,
    save_path: Path,
    max_freq: float = 2500.0,
) -> None:
    """Будує вертикальну 4-панельну систему комплексного зіставлення представлень сигналу.

    Args:
        y (np.ndarray): Відліки часового сигналу.
        sr (int): Частота дискретизації, Гц.
        freqs (np.ndarray): Вектор частот FFT.
        magnitude (np.ndarray): Амплітуди FFT.
        stft_db (np.ndarray): Спектрограма STFT у дБ.
        mel_db (np.ndarray): Мел-спектрограма у дБ.
        hop_length (int): Крок STFT у відліках.
        save_path (Path): Шлях збереження графіка.
        max_freq (float): Максимальна частота для відображення (Гц).
    """
    duration = len(y) / float(sr)
    t = np.arange(len(y)) / float(sr)
    freq_mask = freqs <= max_freq

    fig = plt.figure(figsize=(11.5, 14))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 0.028], wspace=0.04, hspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    cax2 = fig.add_subplot(gs[2, 1])
    cax3 = fig.add_subplot(gs[3, 1])

    # 1. Waveform
    ax0.plot(t, y, color="#1f77b4", linewidth=0.5)
    ax0.set_title("1. Часова форма (Waveform): локалізація амплітуди в часі", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Час, с")
    ax0.set_ylabel("Амплітуда")
    ax0.set_xlim(0.0, duration)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # 2. FFT Spectrum
    ax1.plot(freqs[freq_mask], magnitude[freq_mask], color="#2ca02c", linewidth=0.7)
    ax1.set_title("2. Спектр Фур'є (FFT): частотний склад без часової прив'язки", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Частота, Гц")
    ax1.set_ylabel("Амплітуда |X(f)|")
    ax1.set_xlim(0.0, max_freq)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 3. Spectrogram (STFT)
    img3 = librosa.display.specshow(
        stft_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax2, cmap="magma"
    )
    ax2.set_ylim(0, max_freq)
    ax2.set_xlim(0.0, duration)
    ax2.set_title("3. Спектрограма (STFT): спільний розподіл енергії у часі та частоті", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Час, с")
    ax2.set_ylabel("Частота, Гц")
    fig.colorbar(img3, cax=cax2, format="%+2.0f дБ")

    # 4. Mel-Spectrogram
    img4 = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", htk=True, ax=ax3, cmap="viridis"
    )
    ax3.set_xlim(0.0, duration)
    ax3.set_title("4. Мел-спектрограма: психоакустична шкала HTK, наближена до слуху людини", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Час, с")
    ax3.set_ylabel("Частота (Мел-шкала)")
    fig.colorbar(img4, cax=cax3, format="%+2.0f дБ")

    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Зведену панель порівняння збережено: {save_path}")


def plot_research_n_fft(
    db_dict: dict[int, np.ndarray],
    sr: int,
    hop_length: int,
    save_path: Path,
    max_freq: float = 2500.0,
) -> None:
    """Будує порівняння впливу розміру вікна n_fft на роздільну здатність.

    Args:
        db_dict (dict[int, np.ndarray]): Словник dB-спектрограм за значеннями n_fft.
        sr (int): Частота дискретизації, Гц.
        hop_length (int): Фіксований крок зміщення, відліки.
        save_path (Path): Шлях збереження графіка.
        max_freq (float): Верхня межа частоти для відображення, Гц.
    """
    n_items = len(db_dict)
    fig, axes = plt.subplots(1, n_items, figsize=(5.4 * n_items, 4.6), sharey=True)
    img = None

    for idx, (ax, (n_fft, db_mat)) in enumerate(zip(axes, db_dict.items())):
        delta_f = sr / n_fft
        delta_t_ms = (n_fft / sr) * 1000.0
        img = librosa.display.specshow(
            db_mat, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )
        ax.set_ylim(0, max_freq)
        ax.set_title(
            f"n_fft = {n_fft}\n(Δf = {delta_f:.1f} Гц, Δt = {delta_t_ms:.1f} мс)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Час, с")
        ax.set_ylabel("Частота, Гц" if idx == 0 else "")

    fig.suptitle(
        f"Вплив розміру вікна n_fft на спектрограму (фіксовано hop_length = {hop_length})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.91, top=0.82, bottom=0.14, wspace=0.08)
    if img is not None:
        cbar_ax = fig.add_axes([0.925, 0.14, 0.015, 0.68])
        fig.colorbar(img, cax=cbar_ax, format="%+2.0f дБ")
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік дослідження n_fft збережено: {save_path}")


def plot_research_hop_length(
    db_dict: dict[int, np.ndarray],
    sr: int,
    n_fft: int,
    save_path: Path,
    max_freq: float = 2500.0,
) -> None:
    """Будує порівняння впливу кроку hop_length на часову дискретність спектрограми.

    Args:
        db_dict (dict[int, np.ndarray]): Словник dB-спектрограм за значеннями hop_length.
        sr (int): Частота дискретизації, Гц.
        n_fft (int): Фіксований розмір вікна FFT.
        save_path (Path): Шлях збереження графіка.
        max_freq (float): Верхня межа частоти для відображення, Гц.
    """
    n_items = len(db_dict)
    fig, axes = plt.subplots(1, n_items, figsize=(5.4 * n_items, 4.6), sharey=True)
    img = None

    for idx, (ax, (hop, db_mat)) in enumerate(zip(axes, db_dict.items())):
        overlap_pct = (n_fft - hop) / n_fft * 100.0
        img = librosa.display.specshow(
            db_mat, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )
        ax.set_ylim(0, max_freq)
        ax.set_title(
            f"hop_length = {hop}\n(перекриття {overlap_pct:.1f}%, кадрів: {db_mat.shape[1]})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Час, с")
        ax.set_ylabel("Частота, Гц" if idx == 0 else "")

    fig.suptitle(
        f"Вплив кроку hop_length на спектрограму (фіксовано n_fft = {n_fft})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.91, top=0.82, bottom=0.14, wspace=0.08)
    if img is not None:
        cbar_ax = fig.add_axes([0.925, 0.14, 0.015, 0.68])
        fig.colorbar(img, cax=cbar_ax, format="%+2.0f дБ")
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік дослідження hop_length збережено: {save_path}")


def plot_research_window(
    db_dict: dict[str, np.ndarray],
    sr: int,
    n_fft: int,
    hop_length: int,
    save_path: Path,
    max_freq: float = 2500.0,
) -> None:
    """Будує порівняння впливу форми віконної функції на спектральний витік.

    Args:
        db_dict (dict[str, np.ndarray]): Словник dB-спектрограм за назвами вікон.
        sr (int): Частота дискретизації, Гц.
        n_fft (int): Розмір вікна FFT.
        hop_length (int): Крок зміщення вікна.
        save_path (Path): Шлях збереження графіка.
        max_freq (float): Верхня межа частоти для відображення, Гц.
    """
    n_items = len(db_dict)
    fig, axes = plt.subplots(1, n_items, figsize=(4.8 * n_items, 4.6), sharey=True)
    img = None

    for idx, (ax, (win_name, db_mat)) in enumerate(zip(axes, db_dict.items())):
        img = librosa.display.specshow(
            db_mat, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )
        ax.set_ylim(0, max_freq)
        ax.set_title(f"Вікно: '{win_name}'", fontsize=11, fontweight="bold")
        ax.set_xlabel("Час, с")
        ax.set_ylabel("Частота, Гц" if idx == 0 else "")

    fig.suptitle(
        f"Вплив вагових віконних функцій на спектрограму (n_fft = {n_fft}, hop_length = {hop_length})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.91, top=0.84, bottom=0.14, wspace=0.08)
    if img is not None:
        cbar_ax = fig.add_axes([0.925, 0.14, 0.015, 0.70])
        fig.colorbar(img, cax=cbar_ax, format="%+2.0f дБ")
    fig.savefig(str(save_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Графік дослідження віконних функцій збережено: {save_path}")
