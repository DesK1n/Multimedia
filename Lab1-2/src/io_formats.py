"""Модуль завантаження зображень та дослідження форматів збереження даних (BMP, PNG, JPG)."""

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import IMAGES_DIR


def robust_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Надійно зчитує зображення на Windows навіть за наявності кирилиці в шляху.

    Args:
        path (str | Path): Шлях до файлу.
        flags (int): Прапорець завантаження OpenCV (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE тощо).

    Returns:
        Optional[np.ndarray]: Завантажене зображення або None при помилці.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        return None
    try:
        data_bytes = np.fromfile(str(path_obj), dtype=np.uint8)
        img = cv2.imdecode(data_bytes, flags)
        return img
    except (OSError, cv2.error) as err:
        print(f"[ERROR] Помилка зчитування {path_obj}: {err}")
        return None


def robust_imwrite(
    path: str | Path, img: np.ndarray, params: Optional[list[int]] = None
) -> bool:
    """Надійно записує зображення на Windows у шляхи з кириличними символами.

    Args:
        path (str | Path): Шлях для збереження.
        img (np.ndarray): Матриця зображення.
        params (Optional[list[int]]): Додаткові параметри кодування OpenCV (наприклад, якість JPEG).

    Returns:
        bool: True при успішному записі, інакше False.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = path_obj.suffix.lower()
    if not ext:
        ext = ".png"

    encode_params = params if params is not None else []
    success, buffer = cv2.imencode(ext, img, encode_params)
    if not success:
        print(f"[ERROR] Не вдалося закодувати зображення у формат {ext}")
        return False

    with open(path_obj, "wb") as f:
        f.write(buffer)
    return True


def load_source_image(custom_path: Optional[str | Path] = None) -> np.ndarray:
    """Завантажує власне зображення користувача у форматі BGR.

    Вимога Демонстрації ЛР №1-2: використання власного зображення студента
    (заборонено використовувати вбудовані зразки на кшталт skimage.data.astronaut).

    Args:
        custom_path (Optional[str | Path]): Довільний явний шлях до файлу.

    Returns:
        np.ndarray: Зображення у форматі BGR (uint8, 3 канали).

    Raises:
        FileNotFoundError: Якщо файл власного зображення відсутній.
    """
    candidates: list[Path] = []
    if custom_path is not None:
        candidates.append(Path(custom_path))

    for ext in ("my_image.png", "my_image.jpg", "my_image.jpeg", "my_image.bmp"):
        candidates.append(IMAGES_DIR / ext)

    for path in candidates:
        if path.is_file():
            img = robust_imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"[INFO] Завантажено власне зображення студента: {path} (shape={img.shape})")
                return img

    raise FileNotFoundError(
        f"Власне зображення не знайдено у {IMAGES_DIR}. "
        "Вимога ЛР1-2: заборонено використовувати вбудований зразок skimage.data. "
        "Будь ласка, розмістіть власне зображення у my_image.png."
    )


def save_formats_and_compare(
    img_bgr: np.ndarray, output_dir: Path
) -> dict[str, float]:
    r"""Зберігає зображення у форматах BMP, PNG, JPG (Q=90) та порівнює розміри.

    Обчислює теоретичний розмір растра без стиснення:
    $$S_{\text{theo}} = \frac{M \cdot N \cdot c \cdot b}{8 \cdot 1024} \quad [\text{КБ}]$$

    Args:
        img_bgr (np.ndarray): Вхідне BGR зображення.
        output_dir (Path): Каталог для збереження вихідних файлів та звіту.

    Returns:
        dict[str, float]: Словник розмірів файлів (у КБ) та теоретичного розміру.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    bmp_path = output_dir / "demo.bmp"
    png_path = output_dir / "demo.png"
    jpg_path = output_dir / "demo.jpg"

    # BMP: збереження нестисненого растру
    robust_imwrite(bmp_path, img_bgr)
    # PNG: стиснення без втрат (алгоритм Deflate)
    robust_imwrite(png_path, img_bgr)
    # JPG: стиснення з втратами (DCT + квантування коефіцієнтів)
    robust_imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    h, w, c = img_bgr.shape
    bytes_per_channel = 1  # uint8 = 8 біт на канал
    theoretical_bytes = h * w * c * bytes_per_channel
    theoretical_kb = theoretical_bytes / 1024.0

    sizes_kb: dict[str, float] = {
        "Теоретичний розмір (без стиснення)": theoretical_kb,
        "BMP (нестиснений растр)": bmp_path.stat().st_size / 1024.0,
        "PNG (стиснення без втрат)": png_path.stat().st_size / 1024.0,
        "JPG (якість Q=90)": jpg_path.stat().st_size / 1024.0,
    }

    csv_path = output_dir / "01_formats.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Формат / Опис", "Розмір, КБ", "Коефіцієнт стиснення (до теор.)"])
        for name, size in sizes_kb.items():
            ratio = theoretical_kb / size if size > 0 else 1.0
            writer.writerow([name, f"{size:.2f}", f"{ratio:.2f}x"])

    print(f"[INFO] Результати порівняння форматів збережено у: {csv_path}")
    return sizes_kb
