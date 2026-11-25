"""Крок 8. Дослідження залежності швидкодії (FPS) та точності (mAP) від роздільної здатності.

Вимірює частоту обробки кадрів (FPS) та якість детекції (mAP@50, mAP@50-95) на COCO128
для вхідних розмірів зображення: imgsz ∈ {320, 480, 640, 960, 1280}.
"""

import json
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src import config, utils

N_WARMUP: int = 2
N_REPEATS: int = 6


def measure_fps(model: YOLO, image: np.ndarray, imgsz: int) -> float:
    """Вимірює середній FPS інференсу моделі для заданого розміру входу.

    Args:
        model: Модель YOLO.
        image: Зображення для тесту.
        imgsz: Розмірність більшої сторони вхідного кадру.

    Returns:
        float: Кількість оброблених кадрів за секунду (FPS).
    """
    for _ in range(N_WARMUP):
        model.predict(source=image, imgsz=imgsz, conf=0.25, verbose=False)

    start_time = time.perf_counter()
    for _ in range(N_REPEATS):
        model.predict(source=image, imgsz=imgsz, conf=0.25, verbose=False)
    elapsed_time = time.perf_counter() - start_time

    avg_time_per_frame = elapsed_time / N_REPEATS
    return 1.0 / max(1e-6, avg_time_per_frame)


def run_fps_resolution_research() -> List[Dict[str, Any]]:
    """Проводить серію вимірювань FPS та mAP за різної просторової роздільності.

    Returns:
        List[Dict[str, Any]]: Список результатів замірів для кожного imgsz.
    """
    model_path = (
        config.YOLO_NANO_WEIGHTS
        if config.YOLO_NANO_WEIGHTS.exists()
        else "yolov8n.pt"
    )
    model = YOLO(str(model_path))
    image_path = utils.download_demo_image()
    image = utils.robust_imread(image_path)

    records: List[Dict[str, Any]] = []
    print("\nВимірювання швидкодії (FPS) та mAP залежно від розмірності imgsz:")
    for size in config.IMAGE_SIZES:
        fps = measure_fps(model, image, size)
        latency_ms = 1000.0 / max(1e-6, fps)
        val_res = model.val(
            data=config.VAL_DATASET, imgsz=size, verbose=False, plots=False
        )
        map50 = float(val_res.box.map50)
        map50_95 = float(val_res.box.map)
        records.append(
            {
                "imgsz": size,
                "fps": round(fps, 2),
                "latency_ms": round(latency_ms, 2),
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
            }
        )
        print(
            f"  imgsz = {size:<5} -> {fps:5.1f} FPS ({latency_ms:5.1f} мс) | "
            f"mAP@50 = {map50 * 100:.1f}% | mAP@50-95 = {map50_95 * 100:.1f}%"
        )

    sizes = [r["imgsz"] for r in records]
    fps_values = [r["fps"] for r in records]
    map50_vals = [r["map50"] * 100 for r in records]
    map50_95_vals = [r["map50_95"] * 100 for r in records]

    fig, (ax_fps, ax_map) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. FPS = f(imgsz)
    ax_fps.plot(
        sizes,
        fps_values,
        marker="o",
        linewidth=2.5,
        color="#d62728",
        label="Швидкодія (FPS)",
    )
    ax_fps.set_xlabel("Роздільна здатність (imgsz, пікселі)", fontsize=10.5)
    ax_fps.set_ylabel("Швидкість обробки (FPS)", fontsize=10.5)
    ax_fps.set_title(
        "Швидкодія інференсу FPS = f(imgsz)", fontsize=11.5, fontweight="bold"
    )
    ax_fps.set_xticks(sizes)
    ax_fps.set_ylim(0, max(fps_values) * 1.18)
    ax_fps.grid(True, linestyle="--", alpha=0.5)
    ax_fps.legend(fontsize=9.5, loc="upper right")

    for x, y in zip(sizes, fps_values):
        offset = (10, 7) if x in (320, 480) else (0, 9)
        ax_fps.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#d62728",
        )

    # 2. mAP = f(imgsz)
    ax_map.plot(
        sizes,
        map50_vals,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="mAP@50 (%)",
    )
    ax_map.plot(
        sizes,
        map50_95_vals,
        marker="s",
        linewidth=2.2,
        color="#2ca02c",
        label="mAP@50-95 (%)",
    )
    ax_map.set_xlabel("Роздільна здатність (imgsz, пікселі)", fontsize=10.5)
    ax_map.set_ylabel("Точність детекції (%)", fontsize=10.5)
    ax_map.set_title(
        "Точність на COCO128 mAP = f(imgsz)", fontsize=11.5, fontweight="bold"
    )
    ax_map.set_xticks(sizes)
    ax_map.set_ylim(
        max(0.0, min(map50_95_vals) - 10.0),
        min(100.0, max(map50_vals) + 10.0),
    )
    ax_map.grid(True, linestyle="--", alpha=0.5)
    ax_map.legend(fontsize=9.5, loc="lower right")

    for x, m50, m95 in zip(sizes, map50_vals, map50_95_vals):
        ax_map.annotate(
            f"{m50:.1f}%",
            (x, m50),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1f77b4",
        )
        ax_map.annotate(
            f"{m95:.1f}%",
            (x, m95),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#2ca02c",
        )

    plot_file = config.PLOTS_DIR / "step8_fps_vs_resolution.png"
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Графік збережено: {plot_file}")

    out_file = config.DATA_DIR / "step8_fps_resolution.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✔ Результати кроку 8 збережено: {out_file}")
    return records


def main() -> None:
    """Точка входу кроку 8."""
    run_fps_resolution_research()


if __name__ == "__main__":
    main()
