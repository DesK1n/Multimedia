"""Крок 9. Порівняльний аналіз моделей YOLOv8n (Nano) та YOLOv8s (Small).

Оцінює компроміс між точністю (Accuracy / mAP) та швидкодією (FPS)
для двох масштабів архітектури YOLOv8.
"""

import csv
import json
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src import config, utils


def measure_fps(
    model: YOLO, image: np.ndarray, imgsz: int = 640, repeats: int = 6
) -> float:
    """Вимірює середній FPS на стандартному розмірі 640.

    Args:
        model: Модель YOLO.
        image: Зображення для тесту.
        imgsz: Розмір входу.
        repeats: Кількість повторів.

    Returns:
        float: Частота обробки у FPS.
    """
    for _ in range(2):
        model.predict(source=image, imgsz=imgsz, conf=0.25, verbose=False)

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(source=image, imgsz=imgsz, conf=0.25, verbose=False)
    elapsed = time.perf_counter() - start
    return repeats / max(1e-6, elapsed)


def evaluate_yolo_model(weights_name: str, image: np.ndarray) -> Dict[str, Any]:
    """Виконує повне тестування моделі YOLOv8 на наборі COCO128.

    Args:
        weights_name: Назва файлу ваг (наприклад, yolov8n.pt або yolov8s.pt).
        image: Тестове зображення для вимірювання швидкодії.

    Returns:
        Dict[str, Any]: Метрики параметрів, якості та швидкодії.
    """
    local_weights = config.WEIGHTS_DIR / weights_name
    model = YOLO(str(local_weights) if local_weights.exists() else weights_name)
    if not local_weights.exists():
        import shutil
        from pathlib import Path

        def_downloaded = Path(weights_name)
        if def_downloaded.exists():
            shutil.move(str(def_downloaded), str(local_weights))

    num_params = sum(p.numel() for p in model.model.parameters())
    print(
        f"\n--- Оцінка моделі {weights_name} ({num_params:,} параметрів) ---"
    )

    metrics = model.val(data=config.VAL_DATASET, verbose=False)
    fps = measure_fps(model, image, imgsz=640)

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print(
        f"  Precision: {precision:.3f} | Recall: {recall:.3f} | "
        f"mAP@50: {map50:.3f} | mAP@50-95: {map50_95:.3f} | FPS: {fps:.1f}"
    )

    return {
        "model_name": weights_name,
        "parameters": num_params,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "fps": round(fps, 2),
        "latency_ms": round(1000.0 / max(1e-6, fps), 2),
    }


def plot_model_comparison(
    models_data: List[Dict[str, Any]], save_path: str
) -> None:
    """Будує порівняльні діаграми для Nano та Small моделей.

    Args:
        models_data: Список результатів тестування моделей.
        save_path: Шлях для збереження зображення.
    """
    names = [m["model_name"] for m in models_data]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Метрики mAP
    ax = axes[0]
    map50 = [m["map50"] * 100 for m in models_data]
    map50_95 = [m["map50_95"] * 100 for m in models_data]
    width = 0.35
    b_m50 = ax.bar(x - width / 2, map50, width, label="mAP@50", color="#1f77b4")
    b_m95 = ax.bar(
        x + width / 2, map50_95, width, label="mAP@50-95", color="#aec7e8"
    )
    for b in list(b_m50) + list(b_m95):
        y_val = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y_val + 1.2,
            f"{y_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_title("Якість детекції (mAP)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Значення (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 2. Кількість параметрів
    ax = axes[1]
    params = [m["parameters"] / 1_000_000.0 for m in models_data]
    bars_p = ax.bar(names, params, color="#2ca02c", width=0.45)
    for b in bars_p:
        y_val = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y_val + max(params) * 0.02,
            f"{y_val:.2f}M",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title(
        "Параметри моделі (млн ваг)", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Мільйонів параметрів", fontsize=10)
    ax.set_ylim(0, max(params) * 1.18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 3. Швидкодія (FPS)
    ax = axes[2]
    fps = [m["fps"] for m in models_data]
    bars_f = ax.bar(names, fps, color="#d62728", width=0.45)
    for b in bars_f:
        y_val = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y_val + max(fps) * 0.02,
            f"{y_val:.1f} FPS",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title(
        "Швидкодія інференсу (FPS, 640px)", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Кадрів / сек", fontsize=10)
    ax.set_ylim(0, max(fps) * 1.18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_model_comparison() -> Dict[str, Any]:
    """Виконує порівняння моделей YOLOv8n та YOLOv8s.

    Returns:
        Dict[str, Any]: Словник із даними порівняння.
    """
    image_path = utils.download_demo_image()
    image = utils.robust_imread(image_path)

    results: Dict[str, Any] = {}
    for weights_file in ("yolov8n.pt", "yolov8s.pt"):
        results[weights_file] = evaluate_yolo_model(weights_file, image)

    models_list = list(results.values())
    plot_path = config.PLOTS_DIR / "step9_model_comparison.png"
    plot_model_comparison(models_list, str(plot_path))
    print(f"✔ Порівняльний графік збережено: {plot_path}")

    # Збереження CSV таблиці
    csv_file = config.DATA_DIR / "step9_model_comparison.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(models_list[0].keys()))
        writer.writeheader()
        writer.writerows(models_list)
    print(f"✔ Порівняльну таблицю збережено: {csv_file}")

    out_file = config.DATA_DIR / "step9_model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✔ Результати кроку 9 збережено: {out_file}")
    return results


def main() -> None:
    """Точка входу кроку 9."""
    run_model_comparison()


if __name__ == "__main__":
    main()
