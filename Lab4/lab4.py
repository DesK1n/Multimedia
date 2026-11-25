"""Головний модуль запуску лабораторної роботи № 4.

Виконує 4 контрольовані експерименти з класифікації Fashion-MNIST:
- Experiment A: Custom CNN без аугментації.
- Experiment B: Custom CNN з аугментацією.
- Experiment C: Transfer Learning на базі ResNet-18.
- Experiment D: Transfer Learning на базі EfficientNet-B0.

Зберігає порівняльні таблиці у CSV та JSON, будує матриці невідповідностей,
криві навчання та підсумкові порівняльні діаграми.
"""

import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from src import config, experiments, plotting


def set_seed(seed: int) -> None:
    """Фіксує псевдовипадкові генератори для повної відтворюваності.

    Args:
        seed: Числове зерно ініціалізації.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_comparison_table(
    results: List[experiments.ExperimentResult],
) -> List[Dict[str, Any]]:
    """Формує структуровану порівняльну таблицю результатів.

    Args:
        results: Список результатів усіх експериментів.

    Returns:
        List[Dict[str, Any]]: Рядки порівняльної таблиці для експорту в CSV.
    """
    table = []
    for exp in results:
        table.append(
            {
                "Experiment": exp.title,
                "Short_Name": exp.short_name,
                "Accuracy": round(exp.eval_result.accuracy, 4),
                "Precision": round(exp.eval_result.precision, 4),
                "Recall": round(exp.eval_result.recall, 4),
                "F1_Weighted": round(exp.eval_result.f1, 4),
                "F1_Macro": round(exp.eval_result.f1_macro, 4),
                "Total_Params": exp.total_params,
                "Trainable_Params": exp.trainable_params,
                "Memory_MB": round(exp.memory_mb, 2),
                "Training_Time_s": round(exp.history.training_time_sec, 2),
                "Inference_Time_s": round(
                    exp.eval_result.inference_time_sec, 3
                ),
                "Batch_Latency_ms": round(
                    exp.eval_result.batch_latency_ms, 2
                ),
            }
        )
    return table


def print_comparison_table(table: List[Dict[str, Any]]) -> None:
    """Виводить відформатовану таблицю результатів у стандартний потік виводу.

    Args:
        table: Підготовлений список словників метрик.
    """
    header = (
        f"{'Експеримент':<32} | {'Acc (%)':>7} | {'Prec (%)':>8} | "
        f"{'Rec (%)':>7} | {'F1 (%)':>7} | {'Всього ваг':>11} | "
        f"{'Навч. ваг':>10} | {'Пам. (MB)':>9} | {'Час (с)':>7} | {'Lat (мс)':>8}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("ПІДСУМКОВА ПОРІВНЯЛЬНА ТАБЛИЦЯ ЕКСПЕРИМЕНТІВ")
    print(separator)
    print(header)
    print(separator)

    for row in table:
        print(
            f"{row['Short_Name']:<32} | "
            f"{row['Accuracy']*100:7.2f} | "
            f"{row['Precision']*100:8.2f} | "
            f"{row['Recall']*100:7.2f} | "
            f"{row['F1_Weighted']*100:7.2f} | "
            f"{row['Total_Params']:11,d} | "
            f"{row['Trainable_Params']:10,d} | "
            f"{row['Memory_MB']:9.2f} | "
            f"{row['Training_Time_s']:7.1f} | "
            f"{row['Batch_Latency_ms']:8.2f}"
        )
    print(separator + "\n")


def main() -> int:
    """Головна точка входу для запуску лабораторної роботи № 4.

    Returns:
        int: Код завершення програми (0 при успішному виконанні).
    """
    set_seed(config.SEED)

    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА № 4: ОБРОБКА МУЛЬТИМЕДІЙНИХ ДАНИХ")
    print("ТЕМА: CNN ТА TRANSFER LEARNING ДЛЯ КЛАСИФІКАЦІЇ МУЛЬТИМЕДІА")
    print("=" * 80)
    print(f"Обчислювальний пристрій: {config.DEVICE}")
    print(f"Кількість класів: {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(
        f"Розмір підвибірок: {config.TRAIN_PER_CLASS * config.NUM_CLASSES} "
        f"навчальних, {config.VAL_PER_CLASS * config.NUM_CLASSES} валідаційних, "
        f"{config.TEST_PER_CLASS * config.NUM_CLASSES} тестових зразків"
    )
    print(f"Кількість епох: {config.EPOCHS} | Розмір батчу: {config.BATCH_SIZE}")

    results = experiments.run_all_experiments()
    ablation_rows = experiments.run_augmentation_ablation(results)

    table = build_comparison_table(results)
    print_comparison_table(table)

    # 1. Збереження результатів у CSV
    csv_file: Path = config.TABLES_DIR / "comparison_table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f"✔ Таблицю результатів збережено: {csv_file}")

    ablation_csv: Path = config.TABLES_DIR / "augmentation_ablation.csv"
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"✔ Таблицю абляції аугментацій збережено: {ablation_csv}")

    # 2. Збереження зведеного JSON (із поепоховою історією)
    histories = {
        exp.key: {
            "short_name": exp.short_name,
            "train_loss": [round(float(v), 4) for v in exp.history.train_loss],
            "train_accuracy": [round(float(v), 4) for v in exp.history.train_accuracy],
            "val_loss": [round(float(v), 4) for v in exp.history.val_loss],
            "val_accuracy": [round(float(v), 4) for v in exp.history.val_accuracy],
        }
        for exp in results
    }
    summary = {
        "device": str(config.DEVICE),
        "seed": config.SEED,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "train_per_class": config.TRAIN_PER_CLASS,
        "val_per_class": config.VAL_PER_CLASS,
        "test_per_class": config.TEST_PER_CLASS,
        "table": table,
        "histories": histories,
        "augmentation_ablation": ablation_rows,
    }
    json_file: Path = config.TABLES_DIR / "summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✔ Зведений JSON збережено: {json_file}")

    # 3. Побудова графіків
    cm_path: Path = config.FIGURES_DIR / "confusion_matrices.png"
    lc_path: Path = config.FIGURES_DIR / "learning_curves.png"
    cb_path: Path = config.FIGURES_DIR / "comparison_bars.png"
    ab_path: Path = config.FIGURES_DIR / "augmentation_ablation.png"

    plotting.plot_confusion_matrices(results, cm_path)
    plotting.plot_learning_curves(results, lc_path)
    plotting.plot_comparison_bars(results, cb_path)
    plotting.plot_augmentation_ablation(ablation_rows, ab_path)

    print(f"✔ Графіки збережено у каталозі: {config.FIGURES_DIR}")
    print("\nУсі експерименти успішно завершено!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
