"""Модуль математичних обчислень FFT, STFT та Mel-спектрограм."""

import librosa
import numpy as np
import scipy.fft


def compute_fft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Обчислює дискретне швидке перетворення Фур'є (FFT) для дійсного сигналу.

    Математична модель:
    $$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}, \quad k = 0, \dots, \lfloor N/2 \rfloor$$

    Args:
        y (np.ndarray): Одновимірний масив відліків сигналу.
        sr (int): Частота дискретизації, Гц.

    Returns:
        tuple[np.ndarray, np.ndarray]: Вектор додатних частот (Гц) та вектор амплітуд $|X(f)|$.
    """
    n = len(y)
    spectrum = scipy.fft.rfft(y)
    freqs = scipy.fft.rfftfreq(n, d=1.0 / sr)
    magnitude = np.abs(spectrum)
    return freqs, magnitude


def compute_stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
) -> np.ndarray:
    r"""Обчислює короткочасне віконне перетворення Фур'є (STFT).

    Математична модель:
    $$\text{STFT}\{x\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - m \cdot H] \cdot e^{-j \omega n}$$

    Args:
        y (np.ndarray): Вхідний аудіосигнал.
        n_fft (int): Довжина вікна аналізу у відліках.
        hop_length (int): Крок зміщення вікна (hop length) у відліках.
        window (str): Назва вагового вікна ("hann", "hamming", "boxcar", "blackman").

    Returns:
        np.ndarray: Комплексна двовимірна матриця STFT розмірністю (1 + n_fft // 2, n_frames).
    """
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)


def stft_to_db(stft_matrix: np.ndarray) -> np.ndarray:
    """Перетворює комплексну матрицю STFT в логарифмічну амплітудну шкалу децибелів.

    Args:
        stft_matrix (np.ndarray): Комплексна матриця STFT.

    Returns:
        np.ndarray: Логарифмічна матриця амплітуд у дБ (діапазон [-80, 0]).
    """
    magnitude = np.abs(stft_matrix)
    return librosa.amplitude_to_db(magnitude, ref=np.max)


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    r"""Обчислює психоакустичну мел-спектрограму звукового сигналу за шкалою HTK.

    Формула переходу від частоти $f$ (Гц) до шкали Мел $m$:
    $$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

    Args:
        y (np.ndarray): Вхідний аудіосигнал.
        sr (int): Частота дискретизації.
        n_fft (int): Розмір вікна FFT.
        hop_length (int): Крок зміщення вікна.
        n_mels (int): Кількість смуг гребінки мел-фільтрів.

    Returns:
        np.ndarray: Енергетична мел-спектрограма потужності.
    """
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, htk=True
    )


def mel_to_db(mel_spec: np.ndarray) -> np.ndarray:
    """Переводить мел-спектрограму потужності у логарифмічні децибели.

    Args:
        mel_spec (np.ndarray): Матриця мел-спектрограми потужності.

    Returns:
        np.ndarray: Логарифмічна мел-спектрограма у дБ (діапазон [-80, 0]).
    """
    return librosa.power_to_db(mel_spec, ref=np.max)
