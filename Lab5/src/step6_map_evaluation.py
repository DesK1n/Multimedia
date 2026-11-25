"""Крок 6. Оцінювання якості детекції (Precision, Recall, mAP) на COCO128.

Виконує валідацію моделі на розміченому датасеті COCO128 та визначає класи
з найкращими й найгіршими показниками середньої точності AP50.
"""

import json
from typing import Any, Dict

import matplotlib.pyplot as plt
from ultralytics import YOLO

from src import config


def run_map_evaluation() -> Dict[str, Any]:
    """Виконує валідацію на датасеті COCO128 та розраховує mAP.

    Returns:
        Dict[str, Any]: Словник з основними метриками та аналізом класів.
    """
    model_path = (
        config.YOLO_NANO_WEIGHTS
        if config.YOLO_NANO_WEIGHTS.exists()
        else "yolov8n.pt"
    )
    model = YOLO(str(model_path))

    print("Запуск валідації моделі на наборі COCO128...")
    metrics = model.val(data=config.VAL_DATASET, verbose=False)

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print(f"Precision (mean):  {precision:.3f}")
    print(f"Recall (mean):     {recall:.3f}")
    print(f"mAP@50:            {map50:.3f}")
    print(f"mAP@50-95:         {map50_95:.3f}")

    class_ap50 = {}
    for i, cls_id in enumerate(metrics.ap_class_index):
        cls_name = model.names[int(cls_id)]
        class_ap50[cls_name] = float(metrics.box.ap50[i])

    sorted_classes = sorted(class_ap50.items(), key=lambda kv: kv[1])
    worst_5 = sorted_classes[:5]
    best_5 = sorted_classes[-5:][::-1]

    print("\nНайгірші 5 класів за AP50:")
    for name, ap in worst_5:
        print(f"  {name:<15} AP50 = {ap:.3f}")

    print("\nНайкращі 5 класів за AP50:")
    for name, ap in best_5:
        print(f"  {name:<15} AP50 = {ap:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    metric_names = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    metric_vals = [precision * 100, recall * 100, map50 * 100, map50_95 * 100]
    metric_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    bars = ax1.bar(metric_names, metric_vals, color=metric_colors, width=0.55)
    for bar, val in zip(bars, metric_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Значення (%)", fontsize=10)
    ax1.set_title(
        "Інтегральні метрики YOLOv8n на COCO128",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    combined = worst_5 + sorted_classes[-5:]
    cls_labels = [name for name, _ in combined]
    cls_vals = [ap for _, ap in combined]
    cls_colors = ["#d62728"] * len(worst_5) + ["#2ca02c"] * len(best_5)
    hbars = ax2.barh(cls_labels, cls_vals, color=cls_colors, height=0.65)
    for bar, val in zip(hbars, cls_vals):
        ax2.text(
            val + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    ax2.axvline(
        map50,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Середній mAP@50 = {map50:.3f}",
    )
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("AP@50", fontsize=10)
    ax2.set_title(
        "5 найгірших vs 5 найкращих класів (AP@50)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    plot_file = config.PLOTS_DIR / "step6_map_evaluation.png"
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Графік метрик COCO128 збережено: {plot_file}")

    summary = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "worst_classes": [
            {"class": n, "ap50": round(a, 3)} for n, a in worst_5
        ],
        "best_classes": [
            {"class": n, "ap50": round(a, 3)} for n, a in best_5
        ],
        "plot_path": str(plot_file),
    }

    out_file = config.DATA_DIR / "step6_map_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"✔ Результати кроку 6 збережено: {out_file}")
    return summary


def main() -> None:
    """Точка входу кроку 6."""
    run_map_evaluation()


if __name__ == "__main__":
    main()
