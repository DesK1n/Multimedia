"""Крок 3. Дослідження та візуалізація метрики Intersection over Union (IoU).

Демонструє геометричний зміст коефіцієнта IoU на трьох тестових прямокутниках
із різним ступенем просторового перекриття.
"""

import json
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt

from src import config, utils


def draw_box(ax: Any, box: Sequence[float], color: str, label: str) -> None:
    """Малює прямокутник на осях Matplotlib.

    Args:
        ax: Осі графіка Matplotlib.
        box: Координати [x1, y1, x2, y2].
        color: Колір лінії.
        label: Текстова мітка для легенди.
    """
    x1, y1, x2, y2 = box
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=2.5,
            label=label,
        )
    )


def run_iou_experiment() -> Dict[str, Any]:
    """Виконує обчислення та візуалізацію IoU.

    Returns:
        Dict[str, Any]: Словник зі значеннями IoU та координатами боксів.
    """
    box1 = [50.0, 50.0, 150.0, 150.0]
    box2 = [90.0, 90.0, 190.0, 190.0]
    box3 = [300.0, 300.0, 400.0, 400.0]

    iou_12 = utils.calculate_iou(box1, box2)
    iou_13 = utils.calculate_iou(box1, box3)
    iou_11 = utils.calculate_iou(box1, box1)

    print(f"IoU(box1, box2) = {iou_12:.3f} (частковий перетин)")
    print(f"IoU(box1, box3) = {iou_13:.3f} (відсутність перетину)")
    print(f"IoU(box1, box1) = {iou_11:.3f} (самоперетин - еталон)")

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_box(ax, box1, "#1f77b4", "Box 1 [50, 50, 150, 150]")
    draw_box(ax, box2, "#ff7f0e", "Box 2 [90, 90, 190, 190]")
    draw_box(ax, box3, "#d62728", "Box 3 [300, 300, 400, 400]")

    # Зафарбовуємо область перетину Box 1 і Box 2
    inter_x1, inter_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    inter_x2, inter_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        ax.add_patch(
            plt.Rectangle(
                (inter_x1, inter_y1),
                inter_x2 - inter_x1,
                inter_y2 - inter_y1,
                fill=True,
                facecolor="green",
                alpha=0.35,
                label=f"Перетин Box 1 і Box 2 (IoU = {iou_12:.3f})",
            )
        )

    ax.text(
        box3[0] + 5,
        box3[1] - 10,
        f"IoU(1, 3) = {iou_13:.3f}",
        color="#d62728",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(0, 440)
    ax.set_ylim(440, 0)  # Інвертуємо вісь Y для екранної орієнтації
    ax.set_title(
        "Геометрична ілюстрація IoU (3 прямокутники)\n"
        f"IoU(1,1)={iou_11:.3f} | IoU(1,2)={iou_12:.3f} | IoU(1,3)={iou_13:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("X координата (пікселі)", fontsize=10)
    ax.set_ylabel("Y координата (пікселі)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_aspect("equal")

    plot_file = config.PLOTS_DIR / "step3_iou_visualization.png"
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Графік збережено: {plot_file}")

    summary = {
        "iou_box1_box2": round(iou_12, 4),
        "iou_box1_box3": round(iou_13, 4),
        "iou_box1_box1": round(iou_11, 4),
        "boxes": {"box1": box1, "box2": box2, "box3": box3},
        "plot_path": str(plot_file),
    }

    out_file = config.DATA_DIR / "step3_iou.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"✔ Результати кроку 3 збережено: {out_file}")
    return summary


def main() -> None:
    """Точка входу кроку 3."""
    run_iou_experiment()


if __name__ == "__main__":
    main()
