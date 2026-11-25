"""Моделі та архітектури нейронних мереж для класифікації зображень.

Містить власну реалізацію згорткової нейромережі (SimpleCNN) та фабричні
функції для конфігурації попередньо навчених моделей ResNet-18 і
EfficientNet-B0 у режимі Transfer Learning (Linear Probing).
"""

import torch
import torch.nn as nn
from torchvision import models

from src import config


class SimpleCNN(nn.Module):
    """Власна згорткова нейронна мережа для зображень розміром 28x28.

    Архітектура складається з 2 послідовних згорткових блоків
    (Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D) та повнозв'язного
    класифікатора з шаром регуляризації Dropout.

    Attributes:
        features: Послідовність шарів вилучення ознак.
        classifier: Повнозв'язна голова класифікації.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES) -> None:
        """Ініціалізує шари SimpleCNN.

        Args:
            num_classes: Кількість вихідних класів класифікації.
        """
        super().__init__()

        self.features: nn.Sequential = nn.Sequential(
            # Блок 1: 1 -> 32 канали, 28x28 -> 14x14
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Блок 2: 32 -> 64 канали, 14x14 -> 7x7
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід мережі.

        Args:
            x: Вхідний тензор форми [B, 1, 28, 28].

        Returns:
            torch.Tensor: Ненормовані логіти класів [B, num_classes].
        """
        features_out = self.features(x)
        logits = self.classifier(features_out)
        return logits


def build_resnet18_transfer(
    num_classes: int = config.NUM_CLASSES,
) -> nn.Module:
    """Конфігурує модель ResNet-18 для задачі Transfer Learning.

    Заморожує всі шари екстрактора ознак ImageNet та замінює вихідний
    повнозв'язний шар (fc) новим лінійним класифікатором.

    Args:
        num_classes: Кількість вихідних категорій.

    Returns:
        nn.Module: Модель ResNet-18 із замороженою базою та новим класифікатором.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_efficientnet_b0_transfer(
    num_classes: int = config.NUM_CLASSES,
) -> nn.Module:
    """Конфігурує модель EfficientNet-B0 для задачі Transfer Learning.

    Заморожує параметри згорткової основи та замінює вихідний лінійний
    класифікатор у послідовності classifier[1].

    Args:
        num_classes: Кількість вихідних категорій.

    Returns:
        nn.Module: Модель EfficientNet-B0, готова до навчання в режимі Linear Probing.
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    in_features: int = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
