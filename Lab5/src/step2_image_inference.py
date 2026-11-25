"""Крок 2. Інференс моделі YOLOv8 на статичних зображеннях.

Здійснює детекцію об'єктів на базовому зображенні bus.jpg та на власному
зображенні студента (my_image.png — Червоний корпус КНУ) із порогом 0.25,
вимірює тривалість прямого проходу та зберігає анотоване зображення.
"""

import json
import time
from typing import Any, Dict, List

from ultralytics import YOLO

from src import config, utils


def _infer_single_image(
    model: YOLO, image_path: Any, out_img_path: Any, conf_thres: float = 0.25
) -> Dict[str, Any]:
    """Виконує інференс на одному зображенні та зберігає анотований результат."""
    start_time = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf_thres, verbose=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    result = results[0]
    print(f"Файл {image_path.name}: час інференсу = {elapsed_ms:.1f} мс | об'єктів = {len(result.boxes)}")

    detections: List[Dict[str, Any]] = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append(
            {"class": cls_name, "confidence": round(conf, 3), "box": xyxy}
        )
        print(f"  [{cls_name:<12}] впевненість={conf:.2f} | рамка={xyxy}")

    annotated = result.plot()
    utils.robust_imwrite(out_img_path, annotated, quality=95)
    print(f"✔ Анотоване зображення збережено: {out_img_path}")

    return {
        "source_image": str(image_path),
        "inference_time_ms": round(elapsed_ms, 2),
        "num_detections": len(result.boxes),
        "detections": detections,
        "annotated_image": str(out_img_path),
    }


def run_image_inference() -> Dict[str, Any]:
    """Виконує детекцію на статичних зображеннях (bus.jpg та власному фото КНУ).

    Returns:
        Dict[str, Any]: Словник із деталізованими детекціями та замірами часу.
    """
    model_path = (
        config.YOLO_NANO_WEIGHTS
        if config.YOLO_NANO_WEIGHTS.exists()
        else "yolov8n.pt"
    )
    model = YOLO(str(model_path))
    image_path = utils.download_demo_image()

    bus_res = _infer_single_image(
        model,
        image_path,
        config.IMAGES_DIR / "step2_detection_result.jpg",
    )

    summary: Dict[str, Any] = {
        "inference_time_ms": bus_res["inference_time_ms"],
        "num_detections": bus_res["num_detections"],
        "detections": bus_res["detections"],
        "annotated_image": bus_res["annotated_image"],
    }

    custom_path = config.BASE_DIR.parent / "Lab1-2" / "images" / "my_image.png"
    if custom_path.exists():
        custom_res = _infer_single_image(
            model,
            custom_path,
            config.IMAGES_DIR / "step2_custom_knu_detection.jpg",
        )
        summary["custom_image"] = custom_res

    out_file = config.DATA_DIR / "step2_image_inference.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✔ Результати кроку 2 збережено: {out_file}")
    return summary


def main() -> None:
    """Точка входу кроку 2."""
    run_image_inference()


if __name__ == "__main__":
    main()
