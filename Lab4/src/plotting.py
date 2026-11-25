"""Модуль візуалізації результатів експериментів.

Будує матриці невідповідностей (Confusion Matrices), криві навчання
(Learning Curves) та підсумкові порівняльні діаграми якості та ресурсів.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.experiments import ExperimentResult


def plot_confusion_matrices(
    results: List[ExperimentResult], save_path: Path
) -> None:
    """Будує сітку 2x2 з матрицями невідповідностей для всіх 4 експериментів.

    Args:
        results: Список результатів експериментів [A, B, C, D].
        save_path: Шлях для збереження зображення у форматі PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, exp in zip(axes.flat, results):
        conf = exp.eval_result.confusion
        im = ax.imshow(conf, cmap="Blues", interpolation="nearest")
        ax.set_title(
            f"{exp.short_name} (Acc: {exp.eval_result.accuracy*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Передбачений клас", fontsize=10)
        ax.set_ylabel("Справжній клас", fontsize=10)
        ax.set_xticks(range(config.NUM_CLASSES))
        ax.set_yticks(range(config.NUM_CLASSES))
        ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(config.CLASS_NAMES, fontsize=8)

        max_val = conf.max()
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                val = conf[i, j]
                text_color = "white" if val > max_val / 2.0 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(
    results: List[ExperimentResult], save_path: Path
) -> None:
    """Будує сітку 2x2 кривих навчання та валідації (Loss і Accuracy) за епохами.

    Args:
        results: Список результатів експериментів.
        save_path: Шлях для збереження зображення у форматі PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    groups = [
        (axes[0, 0], axes[0, 1], results[:2], ["#1f77b4", "#ff7f0e"], "Custom CNN (A vs B)"),
        (axes[1, 0], axes[1, 1], results[2:], ["#2ca02c", "#d62728"], "Transfer Learning (C vs D)"),
    ]

    for ax_loss, ax_acc, group_exps, colors, group_title in groups:
        max_epochs = 1
        for exp, color in zip(group_exps, colors):
            epochs_list = list(range(1, len(exp.history.train_loss) + 1))
            max_epochs = max(max_epochs, len(epochs_list))

            ax_loss.plot(
                epochs_list,
                exp.history.train_loss,
                marker="o",
                color=color,
                linewidth=2,
                label=f"{exp.short_name} (Train)",
            )
            ax_acc.plot(
                epochs_list,
                [acc * 100 for acc in exp.history.train_accuracy],
                marker="o",
                color=color,
                linewidth=2,
                label=f"{exp.short_name} (Train)",
            )

            if exp.history.val_loss:
                ax_loss.plot(
                    epochs_list,
                    exp.history.val_loss,
                    marker="s",
                    linestyle="--",
                    color=color,
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"{exp.short_name} (Val)",
                )
                ax_acc.plot(
                    epochs_list,
                    [acc * 100 for acc in exp.history.val_accuracy],
                    marker="s",
                    linestyle="--",
                    color=color,
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"{exp.short_name} (Val)",
                )

        ticks = list(range(1, max_epochs + 1))
        ax_loss.set_title(f"{group_title}: Функція втрат (Loss)", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Номер епохи", fontsize=10)
        ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax_loss.set_xticks(ticks)
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        ax_loss.legend(fontsize=8.5)

        ax_acc.set_title(f"{group_title}: Точність (Accuracy)", fontsize=11, fontweight="bold")
        ax_acc.set_xlabel("Номер епохи", fontsize=10)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
        ax_acc.set_xticks(ticks)
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        ax_acc.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bars(
    results: List[ExperimentResult], save_path: Path
) -> None:
    """Будує стовпчикові діаграми порівняння якості та обчислювальних витрат.

    Включає:
        1. Якісні метрики (Accuracy, Precision, Recall, F1).
        2. Загальну та навчану кількість параметрів (логарифмічна шкала).
        3. Загальний час навчання (секунди).
        4. Затримку інференсу (мс на батч).

    Args:
        results: Список результатів експериментів.
        save_path: Шлях для збереження зображення у форматі PNG.
    """
    labels = [exp.short_name for exp in results]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Якісні метрики
    ax_metrics = axes[0, 0]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1-Score"]
    bar_width = 0.2

    for i, (key, title) in enumerate(zip(metric_keys, metric_titles)):
        values = [getattr(exp.eval_result, key) * 100 for exp in results]
        ax_metrics.bar(
            x + (i - 1.5) * bar_width, values, width=bar_width, label=title
        )

    ax_metrics.set_title(
        "Порівняння метрик якості класифікації", fontsize=11, fontweight="bold"
    )
    ax_metrics.set_ylabel("Значення (%)", fontsize=10)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_metrics.set_ylim(0, 100)
    ax_metrics.legend(fontsize=8)
    ax_metrics.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 2. Кількість параметрів (log scale)
    ax_params = axes[0, 1]
    total_p = [exp.total_params for exp in results]
    trainable_p = [exp.trainable_params for exp in results]

    ax_params.bar(
        x - 0.18, total_p, width=0.35, label="Всього параметрів", color="#3470a3"
    )
    ax_params.bar(
        x + 0.18,
        trainable_p,
        width=0.35,
        label="Параметри, що навчаються",
        color="#f28e2b",
    )
    ax_params.set_yscale("log")
    ax_params.set_title(
        "Параметри моделі (логарифмічна шкала)", fontsize=11, fontweight="bold"
    )
    ax_params.set_ylabel("Кількість параметрів", fontsize=10)
    ax_params.set_xticks(x)
    ax_params.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_params.legend(fontsize=8)
    ax_params.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 3. Час навчання
    ax_train = axes[1, 0]
    train_times = [exp.history.training_time_sec for exp in results]
    bars_t = ax_train.bar(x, train_times, color="#59a14f", width=0.5)
    for bar in bars_t:
        y_val = bar.get_height()
        ax_train.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_val + max(train_times) * 0.02,
            f"{y_val:.1f}с",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_train.set_title(
        f"Загальний час навчання ({config.EPOCHS} епох)", fontsize=11, fontweight="bold"
    )
    ax_train.set_ylabel("Час (секунди)", fontsize=10)
    ax_train.set_xticks(x)
    ax_train.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_train.set_ylim(0, max(train_times) * 1.18)
    ax_train.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 4. Latency інференсу
    ax_lat = axes[1, 1]
    latencies = [exp.eval_result.batch_latency_ms for exp in results]
    bars_l = ax_lat.bar(x, latencies, color="#e15759", width=0.5)
    for bar in bars_l:
        y_val = bar.get_height()
        ax_lat.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_val + max(latencies) * 0.02,
            f"{y_val:.1f}мс",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_lat.set_title(
        "Затримка інференсу на 1 батч", fontsize=11, fontweight="bold"
    )
    ax_lat.set_ylabel("Затримка (мілісекунди)", fontsize=10)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_lat.set_ylim(0, max(latencies) * 1.18)
    ax_lat.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_augmentation_ablation(
    ablation_rows: list[dict[str, object]], save_path: Path
) -> None:
    """Будує діаграму абляції типів аугментації даних для SimpleCNN.

    Args:
        ablation_rows: Список результатів режимів аугментації.
        save_path: Шлях для збереження зображення у форматі PNG.
    """
    labels = [str(r["label"]) for r in ablation_rows]
    accs = [float(r["test_accuracy"]) * 100.0 for r in ablation_rows]
    f1s = [float(r["f1_weighted"]) * 100.0 for r in ablation_rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5))
    b1 = ax.bar(x - width / 2, accs, width, label="Test Accuracy (%)", color="#4e79a7")
    b2 = ax.bar(x + width / 2, f1s, width, label="F1-Score Weighted (%)", color="#f28e2b")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.35,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    low_y = max(0.0, min(accs + f1s) - 6.0)
    high_y = min(100.0, max(accs + f1s) + 4.0)
    ax.set_ylim(low_y, high_y)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylabel("Значення метрики (%)", fontsize=10.5)
    ax.set_title(
        "Абляція впливу типів аугментації на якість SimpleCNN",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
