"""Організація та виконання чотирьох контрольованих експериментів.

Містить сценарії:
    - Experiment A: Custom CNN без аугментації даних (Baseline);
    - Experiment B: Custom CNN з аугментацією даних (Data Augmentation);
    - Experiment C: Transfer Learning (ResNet-18 ImageNet weights);
    - Experiment D: Transfer Learning (EfficientNet-B0 ImageNet weights).
"""

from dataclasses import dataclass
from typing import List

import torch.nn as nn

from src import config, data, engine, models


@dataclass
class ExperimentResult:
    """Зведені результати окремого експерименту.

    Attributes:
        key: Ідентифікатор експерименту ('A', 'B', 'C', 'D').
        title: Повна академічна назва експерименту.
        short_name: Скорочена назва для графіків та таблиць.
        dataset: Використаний датасет та DataLoader'и.
        total_params: Загальна кількість ваг архітектури.
        trainable_params: Кількість навчаних параметрів.
        memory_mb: Розрахунковий розмір ваг у мегабайтах.
        history: Динаміка навчання за епохами.
        eval_result: Метрики якості та швидкодії на тестових даних.
    """

    key: str
    title: str
    short_name: str
    dataset: data.Datasets
    total_params: int
    trainable_params: int
    memory_mb: float
    history: engine.TrainHistory
    eval_result: engine.EvalResult


def run_single_experiment(
    key: str,
    title: str,
    short_name: str,
    model: nn.Module,
    dataset: data.Datasets,
    learning_rate: float,
) -> ExperimentResult:
    """Уніфікований конвеєр виконання одного експерименту.

    Args:
        key: Буквений індекс ('A', 'B', 'C', 'D').
        title: Повна назва експерименту.
        short_name: Коротка мітка для діаграм.
        model: Модель нейронної мережі.
        dataset: Підготовлені завантажувачі даних.
        learning_rate: Крок оптимізатора.

    Returns:
        ExperimentResult: Структура з усіма зафіксованими даними експерименту.
    """
    total_params, trainable_params = engine.count_params(model)
    memory_mb = engine.estimate_memory_mb(total_params)

    print(f"\n{'='*70}")
    print(f"▶ Старт Experiment {key}: {title}")
    print(
        f"  Параметри: всього = {total_params:,} | навчаються = {trainable_params:,} "
        f"| пам'ять ~ {memory_mb:.2f} MB"
    )
    print(f"{'='*70}")

    history = engine.train_model(
        model=model,
        train_loader=dataset.train_loader,
        epochs=config.EPOCHS,
        learning_rate=learning_rate,
        device=config.DEVICE,
        log_prefix=f"[{key}] ",
        val_loader=dataset.val_loader,
    )

    eval_result = engine.evaluate(model, dataset.test_loader, config.DEVICE)

    print(
        f"✔ Завершено [{key}]: Accuracy = {eval_result.accuracy*100:.2f}% | "
        f"F1 = {eval_result.f1*100:.2f}% | Час навчання = {history.training_time_sec:.1f}с | "
        f"Latency = {eval_result.batch_latency_ms:.1f}мс/батч"
    )

    return ExperimentResult(
        key=key,
        title=title,
        short_name=short_name,
        dataset=dataset,
        total_params=total_params,
        trainable_params=trainable_params,
        memory_mb=memory_mb,
        history=history,
        eval_result=eval_result,
    )


def run_experiment_a() -> ExperimentResult:
    """Experiment A: Власна CNN без аугментації даних (Baseline).

    Returns:
        ExperimentResult: Результат базового експерименту.
    """
    config.set_seed(config.SEED)
    dataset = data.build_datasets(
        image_size=config.CNN_IMAGE_SIZE, channels=1, augment=False
    )
    model = models.SimpleCNN(num_classes=config.NUM_CLASSES)
    return run_single_experiment(
        key="A",
        title="Custom CNN (без аугментації)",
        short_name="A: CNN (Raw)",
        model=model,
        dataset=dataset,
        learning_rate=config.LEARNING_RATE_CNN,
    )


def run_experiment_b() -> ExperimentResult:
    """Experiment B: Власна CNN з аугментацією даних (Data Augmentation).

    Returns:
        ExperimentResult: Результат експерименту з аугментацією.
    """
    config.set_seed(config.SEED)
    dataset = data.build_datasets(
        image_size=config.CNN_IMAGE_SIZE, channels=1, augment=True
    )
    model = models.SimpleCNN(num_classes=config.NUM_CLASSES)
    return run_single_experiment(
        key="B",
        title="Custom CNN (з аугментацією)",
        short_name="B: CNN + Aug",
        model=model,
        dataset=dataset,
        learning_rate=config.LEARNING_RATE_CNN,
    )


def run_experiment_c() -> ExperimentResult:
    """Experiment C: Transfer Learning на базі ResNet-18 (ImageNet).

    Returns:
        ExperimentResult: Результат експерименту з ResNet-18.
    """
    config.set_seed(config.SEED)
    dataset = data.build_datasets(
        image_size=config.PRETRAINED_IMAGE_SIZE,
        channels=config.PRETRAINED_CHANNELS,
        augment=False,
    )
    model = models.build_resnet18_transfer(num_classes=config.NUM_CLASSES)
    return run_single_experiment(
        key="C",
        title="Pretrained ResNet-18 (Transfer Learning)",
        short_name="C: ResNet-18",
        model=model,
        dataset=dataset,
        learning_rate=config.LEARNING_RATE_TRANSFER,
    )


def run_experiment_d() -> ExperimentResult:
    """Experiment D: Transfer Learning на базі EfficientNet-B0 (ImageNet).

    Returns:
        ExperimentResult: Результат експерименту з EfficientNet-B0.
    """
    config.set_seed(config.SEED)
    dataset = data.build_datasets(
        image_size=config.PRETRAINED_IMAGE_SIZE,
        channels=config.PRETRAINED_CHANNELS,
        augment=False,
    )
    model = models.build_efficientnet_b0_transfer(
        num_classes=config.NUM_CLASSES
    )
    return run_single_experiment(
        key="D",
        title="Pretrained EfficientNet-B0 (Transfer Learning)",
        short_name="D: EfficientNet-B0",
        model=model,
        dataset=dataset,
        learning_rate=config.LEARNING_RATE_TRANSFER,
    )


def run_all_experiments() -> List[ExperimentResult]:
    """Послідовно виконує всі 4 обов'язкові експерименти.

    Returns:
        List[ExperimentResult]: Список з 4 результатів (A, B, C, D).
    """
    return [
        run_experiment_a(),
        run_experiment_b(),
        run_experiment_c(),
        run_experiment_d(),
    ]


def run_augmentation_ablation(
    base_results: List[ExperimentResult],
) -> list[dict[str, object]]:
    """Проводить абляцію типів аугментації на базі SimpleCNN (для питання №3).

    Args:
        base_results: Результати основних експериментів (A та B перевикористовуються).

    Returns:
        list[dict[str, object]]: Метрики для 5 режимів аугментації.
    """
    print(f"\n{'=' * 70}")
    print("▶ Абляція типів аугментації даних (SimpleCNN)")
    print(f"{'=' * 70}")

    by_key = {exp.key: exp for exp in base_results}
    modes = [
        ("none", "Без аугментації (Exp A)", "A"),
        ("flip", "HorizontalFlip (p=0.5)", None),
        ("rotation", "RandomRotation (±10°)", None),
        ("affine", "RandomAffine (shift 8%)", None),
        ("combined", "Flip + Rotation (Exp B)", "B"),
    ]

    ablation_rows: list[dict[str, object]] = []
    for mode, label, reuse_key in modes:
        if reuse_key and reuse_key in by_key:
            exp = by_key[reuse_key]
            acc = exp.eval_result.accuracy
            f1 = exp.eval_result.f1
            best_val = (
                max(exp.history.val_accuracy)
                if exp.history.val_accuracy
                else exp.history.train_accuracy[-1]
            )
            t_sec = exp.history.training_time_sec
        else:
            config.set_seed(config.SEED)
            dataset = data.build_datasets(
                image_size=config.CNN_IMAGE_SIZE,
                channels=1,
                augment=True,
                aug_mode=mode,
            )
            model = models.SimpleCNN(num_classes=config.NUM_CLASSES)
            history = engine.train_model(
                model=model,
                train_loader=dataset.train_loader,
                epochs=config.EPOCHS,
                learning_rate=config.LEARNING_RATE_CNN,
                device=config.DEVICE,
                log_prefix=f"[Aug:{mode}] ",
                val_loader=dataset.val_loader,
            )
            eval_res = engine.evaluate(model, dataset.test_loader, config.DEVICE)
            acc = eval_res.accuracy
            f1 = eval_res.f1
            best_val = (
                max(history.val_accuracy)
                if history.val_accuracy
                else history.train_accuracy[-1]
            )
            t_sec = history.training_time_sec

        ablation_rows.append(
            {
                "mode": mode,
                "label": label,
                "test_accuracy": round(float(acc), 4),
                "f1_weighted": round(float(f1), 4),
                "best_val_accuracy": round(float(best_val), 4),
                "train_time_sec": round(float(t_sec), 2),
            }
        )
        print(
            f"  * {label:<26} -> Test Acc = {acc * 100:.2f}% | "
            f"F1 = {f1 * 100:.2f}% | Best Val = {best_val * 100:.2f}%"
        )

    return ablation_rows
