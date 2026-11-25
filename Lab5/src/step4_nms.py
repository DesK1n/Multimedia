"""Крок 4. Дослідження алгоритму Non-Maximum Suppression (NMS).

Реалізує та ілюструє придушення дублікатів рамок за допомогою власного
алгоритму NMS, а також досліджує вплив порогу iou у вбудованому детекторі.
"""

import json
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src import config, utils


def demo_synthetic_boxes() -> Dict[str, Any]:
    """Демонструє роботу власного алгоритму NMS на синтетичних кластерах.

    Returns:
        Dict[str, Any]: Статистика рамок до та після фільтрації.
    """
    boxes = np.array(
        [
            [50.0, 50.0, 200.0, 200.0],
            [55.0, 45.0, 205.0, 195.0],
            [60.0, 60.0, 210.0, 210.0],
            [300.0, 50.0, 400.0, 220.0],
            [305.0, 55.0, 405.0, 225.0],
        ],
        dtype=float,
    )
    scores = np.array([0.90, 0.75, 0.60, 0.85, 0.50])

    print(f"До NMS: кількість рамок = {len(boxes)}")
    keep_indices = utils.non_max_suppression(boxes, scores, iou_threshold=0.5)
    print(
        f"Після NMS: кількість рамок = {len(keep_indices)} (індекси: {keep_indices})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    configs = [
        (axes[0], list(range(len(boxes))), "До застосування NMS (5 рамок)"),
        (axes[1], keep_indices, "Після застосування NMS (2 рамки)"),
    ]

    label_positions = {
        0: (boxes[0][0] + 2, boxes[0][1] - 8, "bottom"),
        1: (boxes[1][2] + 4, boxes[1][1] + 18, "center"),
        2: (boxes[2][0] + 4, boxes[2][3] + 16, "top"),
        3: (boxes[3][0] + 2, boxes[3][1] - 8, "bottom"),
        4: (boxes[4][0] + 4, boxes[4][3] + 16, "top"),
    }

    for ax, indices, title in configs:
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            color = "#d62728" if len(indices) > 2 else "#2ca02c"
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                )
            )
            tx, ty, va_align = label_positions[idx]
            ax.text(
                tx,
                ty,
                f"Score: {scores[idx]:.2f}",
                color=color,
                fontsize=8.5,
                fontweight="bold",
                va=va_align,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.0},
            )
        ax.set_xlim(0, 470)
        ax.set_ylim(265, 0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X координата (пікселі)", fontsize=9)
        ax.set_ylabel("Y координата (пікселі)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal")

    plot_path = config.PLOTS_DIR / "step4_nms_synthetic.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Графік збережено: {plot_path}")

    return {
        "before_count": len(boxes),
        "after_count": len(keep_indices),
        "kept_indices": keep_indices,
        "plot_path": str(plot_path),
    }


def demo_builtin_nms() -> Dict[str, Any]:
    """Досліджує залежність кількості рамок від порогу IoU у вбудованому NMS.

    Returns:
        Dict[str, Any]: Словник з кількістю рамок по порогах та шляхом до графіка.
    """
    model_path = (
        config.YOLO_NANO_WEIGHTS
        if config.YOLO_NANO_WEIGHTS.exists()
        else "yolov8n.pt"
    )
    model = YOLO(str(model_path))
    image_path = utils.download_demo_image()

    results_by_threshold: Dict[float, int] = {}
    rendered_panels: list[tuple[float, int, np.ndarray]] = []
    for threshold in (0.3, 0.5, 0.7, 0.9):
        res = model.predict(
            source=str(image_path),
            conf=0.1,
            iou=threshold,
            verbose=False,
        )
        count = len(res[0].boxes)
        results_by_threshold[threshold] = count
        rendered_rgb = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)
        rendered_panels.append((threshold, count, rendered_rgb))
        print(f"  Вбудований NMS iou={threshold:.1f} -> залишено рамок: {count}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 13.5))
    for ax, (threshold, count, img_rgb) in zip(axes.flat, rendered_panels):
        ax.imshow(img_rgb)
        suffix = " — дублікати" if threshold >= 0.9 else ""
        color = "#d62728" if threshold >= 0.9 else "#111111"
        ax.set_title(
            f"NMS IoU = {threshold:.1f} ({count} рамок){suffix}",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        ax.axis("off")

    fig.suptitle(
        "Вплив порогу IoU вбудованого NMS (YOLOv8n, conf=0.1, bus.jpg)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plot_file = config.PLOTS_DIR / "step4_nms_builtin_comparison.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Порівняльний графік вбудованого NMS збережено: {plot_file}")

    return {"counts": results_by_threshold, "plot_path": str(plot_file)}


def run_nms_experiment() -> Dict[str, Any]:
    """Виконує повний комплекс досліджень NMS.

    Returns:
        Dict[str, Any]: Зведені результати кроку 4.
    """
    synthetic_res = demo_synthetic_boxes()
    builtin_res = demo_builtin_nms()

    summary = {
        "synthetic_example": synthetic_res,
        "builtin_nms_by_iou": {
            str(k): v for k, v in builtin_res["counts"].items()
        },
        "builtin_plot_path": builtin_res["plot_path"],
    }

    out_file = config.DATA_DIR / "step4_nms.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"✔ Результати кроку 4 збережено: {out_file}")
    return summary


def main() -> None:
    """Точка входу кроку 4."""
    run_nms_experiment()


if __name__ == "__main__":
    main()
