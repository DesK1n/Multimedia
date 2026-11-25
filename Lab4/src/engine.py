"""Уніфікований рушій навчання та оцінювання нейромережевих моделей.

Забезпечує стандартизований цикл навчання, збір кривих навчання,
підрахунок параметрів, оцінку споживання пам'яті та комплексне обчислення
метрик якості класифікації та продуктивності інференсу.
"""

from dataclasses import dataclass, field
import time
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Обчислює загальну та навчальну кількість параметрів моделі.

    Args:
        model: Досліджувана модель PyTorch.

    Returns:
        Tuple[int, int]: (загальна_кількість_параметрів, кількість_навчаних_параметрів).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


def estimate_memory_mb(total_params: int) -> float:
    r"""Оцінює теоретичний обсяг оперативної пам'яті для зберігання ваг моделі.

    Розрахунок базується на стандартному типі FP32 (4 байти на 1 вагу):
    $$\text{Memory} = \frac{\text{params} \times 4}{10^6} \quad [\text{MB}]$$

    Args:
        total_params: Загальна кількість параметрів моделі.

    Returns:
        float: Оцінка розміру ваг у мегабайтах (MB).
    """
    return total_params * 4.0 / 1_000_000.0


@dataclass
class TrainHistory:
    """Історія динаміки тренування моделі за епохами.

    Attributes:
        train_loss: Значення функції втрат на навчанні на кожній епосі.
        train_accuracy: Точність на навчанні на кожній епосі.
        val_loss: Значення функції втрат на валідації на кожній епосі.
        val_accuracy: Точність на валідації на кожній епосі.
        training_time_sec: Загальна тривалість тренування у секундах.
    """

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    training_time_sec: float = 0.0


def _freeze_frozen_batchnorm(model: nn.Module) -> None:
    """Фіксує статистику шарів BatchNorm у замороженому бекбоні у режимі eval."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    log_prefix: str = "",
    val_loader: DataLoader | None = None,
) -> TrainHistory:
    """Проводить цикл навчання моделі протягом фіксованої кількості епох.

    Args:
        model: Нейромережева модель для оптимізації.
        train_loader: Завантажувач навчальних даних.
        epochs: Кількість повних епох навчання.
        learning_rate: Швидкість навчання оптимізатора Adam.
        device: Обчислювальний пристрій (CPU або GPU).
        log_prefix: Префікс для форматованого виводу в консоль.
        val_loader: Необов'язковий завантажувач валідаційної вибірки.

    Returns:
        TrainHistory: Структурований об'єкт із метриками епох і часом навчання.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

    history = TrainHistory()
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        _freeze_frozen_batchnorm(model)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        history.train_loss.append(epoch_loss)
        history.train_accuracy.append(epoch_accuracy)

        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_images)
                    val_batch_loss = criterion(val_outputs, val_labels)
                    val_running_loss += val_batch_loss.item() * val_images.size(0)
                    val_correct += (val_outputs.argmax(dim=1) == val_labels).sum().item()
                    val_total += val_labels.size(0)

            val_loss = val_running_loss / max(1, val_total)
            val_acc = val_correct / max(1, val_total)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)
            print(
                f"{log_prefix}Епоха {epoch:02d}/{epochs:02d} -> "
                f"Train Loss: {epoch_loss:.4f} (Acc: {epoch_accuracy * 100:.2f}%) | "
                f"Val Loss: {val_loss:.4f} (Acc: {val_acc * 100:.2f}%)"
            )
        else:
            print(
                f"{log_prefix}Епоха {epoch:02d}/{epochs:02d} -> "
                f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy * 100:.2f}%"
            )

    history.training_time_sec = time.perf_counter() - start_time
    return history


@dataclass
class EvalResult:
    """Комплексні результати оцінки моделі на контрольній вибірці.

    Attributes:
        accuracy: Загальна точність (Accuracy).
        precision: Влучність (Precision weighted).
        recall: Повнота (Recall weighted).
        f1: F1-score (weighted).
        f1_macro: F1-score (macro).
        confusion: Матриця невідповідностей $10 \times 10$.
        inference_time_sec: Загальний час інференсу на всій контрольній вибірці.
        batch_latency_ms: Середня затримка обробки одного батчу в мілісекундах.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    f1_macro: float
    confusion: np.ndarray
    inference_time_sec: float
    batch_latency_ms: float


def evaluate(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> EvalResult:
    """Виконує повне тестування моделі та розраховує аналітичні метрики.

    Args:
        model: Модель для тестування.
        test_loader: Завантажувач тестових даних.
        device: Обчислювальний пристрій.

    Returns:
        EvalResult: Об'єкт із розрахованими метриками точності та затримки.
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    batch_latencies = []

    start_total_time = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:
            t0 = time.perf_counter()
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            t1 = time.perf_counter()

            batch_latencies.append((t1 - t0) * 1000.0)
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

    total_inference_time = time.perf_counter() - start_total_time

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_labels)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES))
    )
    avg_batch_latency = float(np.mean(batch_latencies))

    return EvalResult(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        f1_macro=float(f1_macro),
        confusion=conf_matrix,
        inference_time_sec=total_inference_time,
        batch_latency_ms=avg_batch_latency,
    )
