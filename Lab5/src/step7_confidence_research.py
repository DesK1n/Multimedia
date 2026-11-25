"""Крок 7. Дослідження впливу порогу впевненості (Confidence Threshold).

Аналізує залежність метрик Precision та Recall від порогу c ∈ {0.1, 0.25, 0.5, 0.75, 0.9}
і будує графік дослідницької складової.
"""

import json
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src import config


def run_confidence_research() -> List[Dict[str, Any]]:
    """Виконує параметричний аналіз залежності метрик від порогу c.

    Обчислює точні значення Precision(c), Recall(c) та F1(c) через
    інтерполяцію офіційних кривих Precision-Confidence та Recall-Confidence
    Ultralytics (conf=0.001) без спотворення порожніх класів.

    Returns:
        List[Dict[str, Any]]: Список зафіксованих результатів для кожного порогу.
    """
    model_path = (
        config.YOLO_NANO_WEIGHTS
        if config.YOLO_NANO_WEIGHTS.exists()
        else "yolov8n.pt"
    )
    model = YOLO(str(model_path))

    print("\nДослідження впливу порогу впевненості (c) на кривих COCO128:")
    metrics = model.val(
        data=config.VAL_DATASET, conf=0.001, verbose=False, plots=False
    )
    # curves_results[1]=(conf, F1), [2]=(conf, Precision), [3]=(conf, Recall)
    fx, fy = metrics.box.curves_results[1][0], metrics.box.curves_results[1][1]
    px, py = metrics.box.curves_results[2][0], metrics.box.curves_results[2][1]
    rx, ry = metrics.box.curves_results[3][0], metrics.box.curves_results[3][1]

    records: List[Dict[str, Any]] = []
    for c in config.CONFIDENCE_THRESHOLDS:
        prec = float(np.mean([np.interp(c, px, row) for row in py]))
        rec = float(np.mean([np.interp(c, rx, row) for row in ry]))
        f1 = float(np.mean([np.interp(c, fx, row) for row in fy]))
        records.append(
            {
                "confidence": c,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )
        print(
            f"  c = {c:<4} -> Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}"
        )

    confidences = [r["confidence"] for r in records]
    precisions = [r["precision"] * 100 for r in records]
    recalls = [r["recall"] * 100 for r in records]
    f1_scores = [r["f1"] * 100 for r in records]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        confidences,
        precisions,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Точність (Precision)",
    )
    ax.plot(
        confidences,
        recalls,
        marker="s",
        linewidth=2.5,
        color="#ff7f0e",
        label="Повнота (Recall)",
    )
    ax.plot(
        confidences,
        f1_scores,
        marker="^",
        linestyle="--",
        linewidth=2.0,
        color="#2ca02c",
        label="Баланс (F1-Score)",
    )

    ax.set_xlabel("Поріг впевненості (Confidence Threshold, c)", fontsize=11)
    ax.set_ylabel("Значення метрики (%)", fontsize=11)
    ax.set_title(
        "Вплив порогу впевненості на Precision, Recall та F1 (COCO128)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0.05, 0.96)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9.5, loc="lower left")

    # Підписи точок без накладання
    for x, p, r in zip(confidences, precisions, recalls):
        p_offset = (0, -14) if x >= 0.75 else (0, 8)
        r_offset = (0, 8) if x == 0.1 else (0, -14)
        ax.annotate(
            f"{p:.1f}%",
            (x, p),
            textcoords="offset points",
            xytext=p_offset,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1f77b4",
        )
        ax.annotate(
            f"{r:.1f}%",
            (x, r),
            textcoords="offset points",
            xytext=r_offset,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#ff7f0e",
        )

    plot_file = config.PLOTS_DIR / "step7_precision_recall_vs_confidence.png"
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Графік збережено: {plot_file}")

    out_file = config.DATA_DIR / "step7_confidence_research.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✔ Результати кроку 7 збережено: {out_file}")
    return records


def main() -> None:
    """Точка входу кроку 7."""
    run_confidence_research()


if __name__ == "__main__":
    main()
