"""Допоміжні алгоритмічні та утилітарні функції для детекції об'єктів.

Містить функції завантаження зразків, геометричного розрахунку метрики
Intersection over Union (IoU), власну реалізацію алгоритму Non-Maximum
Suppression (NMS), а також надійні методи роботи з графічними файлами.
"""

from pathlib import Path
from typing import List, Optional, Sequence
import urllib.request

import cv2
import numpy as np

from src import config


def robust_imread(path: Path) -> np.ndarray:
    """Зчитує зображення з диска з коректною підтримкою шляхів Unicode/Cyrillic.

    Args:
        path: Шлях до файлу зображення.

    Returns:
        np.ndarray: Зображення у форматі BGR (uint8).

    Raises:
        FileNotFoundError: Якщо файл не знайдено або не вдалося декодувати.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл не знайдено: {path_obj}")

    data = np.fromfile(str(path_obj), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не вдалося декодувати зображення: {path_obj}")
    return image


def robust_imwrite(
    path: Path, image: np.ndarray, quality: int = 95
) -> bool:
    """Зберігає зображення на диск з коректною підтримкою шляхів Unicode/Cyrillic.

    Args:
        path: Шлях для збереження файлу.
        image: Зображення у форматі BGR.
        quality: Якість стиснення JPEG (1-100).

    Returns:
        bool: True у разі успішного запису.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = path_obj.suffix.lower()

    params = []
    if ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        return False

    with open(path_obj, "wb") as f:
        f.write(buffer)
    return True


def download_if_missing(url: str, path: Path) -> Path:
    """Завантажує файл за посиланням URL, якщо він відсутній локально.

    Args:
        url: Веб-посилання на ресурс.
        path: Локальний шлях для збереження.

    Returns:
        Path: Локальний шлях до збереженого файлу.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            path_obj.write_bytes(response.read())
    return path_obj


def download_demo_image() -> Path:
    """Завантажує стандартне демонстраційне зображення bus.jpg.

    Returns:
        Path: Шлях до збереженого зображення bus.jpg.
    """
    return download_if_missing(
        config.DEMO_IMAGE_URL, config.DEMO_IMAGE_PATH
    )


def download_demo_video() -> Path:
    """Завантажує реальне тестове відео для кроку обробки відеопотоку.

    Returns:
        Path: Шлях до збереженого відеофайлу vtest.avi.
    """
    return download_if_missing(
        config.DEMO_VIDEO_URL, config.DEMO_VIDEO_PATH
    )


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    r"""Обчислює метрику перетину над об'єднанням (Intersection over Union).

    Бокси задаються у форматі $[x_1, y_1, x_2, y_2]$ (лівий верхній та правий
    нижній кути прямокутника):
    $$IoU = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

    Args:
        box_a: Координати першого боксу [x1, y1, x2, y2].
        box_b: Координати другого боксу [x1, y1, x2, y2].

    Returns:
        float: Значення коефіцієнта IoU в діапазоні $[0.0, 1.0]$.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Координати прямокутника перетину
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> List[int]:
    """Власна алгоритмічна реалізація придушення немаксимумів (NMS).

    Фільтрує надлишкові перекриваючися обмежувальні рамки, залишаючи
    лише один бокс із найвищою оцінкою впевненості у кожному кластері.

    Args:
        boxes: Масив рамок розмірності [N, 4] у форматі [x1, y1, x2, y2].
        scores: Масив оцінок впевненості розмірності [N].
        iou_threshold: Поріг коефіцієнта IoU для придушення дублікатів.

    Returns:
        List[int]: Список індексів відібраних рамок у порядку спадання score.
    """
    if len(boxes) == 0:
        return []

    order = list(np.argsort(scores)[::-1])
    keep: List[int] = []

    while order:
        current_idx = int(order.pop(0))
        keep.append(current_idx)

        remaining_indices = []
        for candidate_idx in order:
            iou = calculate_iou(boxes[current_idx], boxes[candidate_idx])
            if iou <= iou_threshold:
                remaining_indices.append(candidate_idx)
        order = remaining_indices

    return keep


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Малює обмежувальні рамки та мітки класів на копії зображення.

    Args:
        image: Вхідне зображення BGR.
        boxes: Масив рамок [N, 4].
        labels: Список текстових назв класів.
        scores: Необов'язковий список оцінок достовірності.
        color: Колір контуру прямокутника (B, G, R).

    Returns:
        np.ndarray: Анотоване зображення.
    """
    annotated = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_text = labels[i]
        if scores is not None:
            label_text = f"{label_text} {scores[i]:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - 6)),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
