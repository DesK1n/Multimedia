"""Крок 5. Обробка реального відеопотоку та покадрова детекція об'єктів.

Завантажує реальне тестове відео OpenCV (vtest.avi), зчитує послідовність кадрів,
здійснює покадрову детекцію YOLOv8n, вимірює FPS обробки та зберігає
анотований відеофайл і контрольні кадри.
"""

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src import config, utils

MAX_FRAMES: int = 90
DEFAULT_FPS: float = 15.0


def read_video_frames(
    video_path: Path, max_frames: int = MAX_FRAMES
) -> Tuple[List[np.ndarray], float]:
    """Зчитує кадри з відеофайлу за допомогою OpenCV VideoCapture.

    Args:
        video_path: Шлях до вхідного відеофайлу.
        max_frames: Максимальна кількість кадрів для зчитування.

    Returns:
        Tuple[List[np.ndarray], float]: Список кадрів BGR та частота кадрів відео.

    Raises:
        RuntimeError: Якщо не вдалося відкрити відеофайл або зчитати жодного кадру.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити відеофайл: {video_path}")

    raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_fps = raw_fps if raw_fps > 0.0 else DEFAULT_FPS

    frames: List[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Відеофайл не містить кадрів: {video_path}")

    return frames, source_fps


def run_video_processing() -> Dict[str, Any]:
    """Завантажує реальне відео, виконує покадрову детекцію YOLOv8n і зберігає звіт.

    Returns:
        Dict[str, Any]: Метрики швидкодії, розподіл класів та шляхи до артефактів.
    """
    video_path = utils.download_demo_video()
    frames, source_fps = read_video_frames(video_path, max_frames=MAX_FRAMES)
    height, width = frames[0].shape[:2]
    print(
        f"Завантажено реальне відео {video_path.name}: "
        f"{len(frames)} кадрів, {width}x{height} @ {source_fps:.2f} FPS"
    )

    model_path = (
        config.YOLO_NANO_WEIGHTS
        if config.YOLO_NANO_WEIGHTS.exists()
        else "yolov8n.pt"
    )
    model = YOLO(str(model_path))

    video_out_path = config.OUTPUT_VIDEO_PATH
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_out_path), fourcc, source_fps, (width, height)
    )

    screenshot_map = {
        0: config.IMAGES_DIR / "step5_video_frame_start.jpg",
        len(frames) // 2: config.IMAGES_DIR / "step5_video_frame_mid.jpg",
        len(frames) - 1: config.IMAGES_DIR / "step5_video_frame_end.jpg",
    }
    screenshot_paths: List[str] = []
    class_counter: Counter[str] = Counter()
    total_detections = 0

    start_time = time.perf_counter()
    try:
        for idx, frame in enumerate(frames):
            results = model.predict(source=frame, conf=0.25, verbose=False)
            boxes = results[0].boxes
            total_detections += len(boxes)

            for cls_id in boxes.cls.tolist():
                cls_name = str(model.names[int(cls_id)])
                class_counter[cls_name] += 1

            annotated_frame = results[0].plot()
            writer.write(annotated_frame)

            if idx in screenshot_map:
                shot_file = screenshot_map[idx]
                utils.robust_imwrite(shot_file, annotated_frame, quality=95)
                screenshot_paths.append(str(shot_file))
    finally:
        writer.release()

    elapsed_time = time.perf_counter() - start_time
    processing_fps = len(frames) / elapsed_time
    avg_detections = total_detections / len(frames)

    print(
        f"Опрацьовано {len(frames)} кадрів за {elapsed_time:.2f}с -> "
        f"швидкодія: {processing_fps:.1f} FPS"
    )
    print(
        f"Всього детекцій: {total_detections} "
        f"(середньо {avg_detections:.2f} на кадр) | Класи: {dict(class_counter)}"
    )
    print(f"✔ Відео збережено: {video_out_path}")

    summary: Dict[str, Any] = {
        "source_video": video_path.name,
        "source_url": config.DEMO_VIDEO_URL,
        "resolution": [width, height],
        "source_fps": round(source_fps, 2),
        "n_frames": len(frames),
        "elapsed_seconds": round(elapsed_time, 2),
        "processing_fps": round(processing_fps, 1),
        "total_detections": total_detections,
        "avg_detections_per_frame": round(avg_detections, 2),
        "class_counts": dict(class_counter),
        "output_video": str(video_out_path),
        "screenshots": screenshot_paths,
    }

    out_file = config.DATA_DIR / "step5_video_processing.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"✔ Результати кроку 5 збережено: {out_file}")
    return summary


def main() -> None:
    """Точка входу кроку 5."""
    run_video_processing()


if __name__ == "__main__":
    main()
