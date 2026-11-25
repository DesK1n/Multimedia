"""Головний модуль запуску лабораторної роботи № 5.

Послідовно виконує всі 9 етапів дослідження об'єктної детекції YOLOv8:
- Крок 1: Налаштування середовища та завантаження моделей.
- Крок 2: Інференс на статичному зображенні bus.jpg.
- Крок 3: Розрахунок та візуалізація метрики IoU.
- Крок 4: Алгоритм Non-Maximum Suppression (власний та вбудований).
- Крок 5: Обробка синтетичного відеопотоку (Ken Burns zoom).
- Крок 6: Оцінка Precision, Recall та mAP на наборі COCO128.
- Крок 7: Дослідження впливу порогу впевненості (c ∈ {0.1..0.9}).
- Крок 8: Дослідження залежності швидкодії (FPS) від роздільної здатності.
- Крок 9: Порівняльний аналіз моделей YOLOv8n та YOLOv8s.
"""

import sys
import time

from src import (
    step1_setup,
    step2_image_inference,
    step3_iou,
    step4_nms,
    step5_video_processing,
    step6_map_evaluation,
    step7_confidence_research,
    step8_fps_resolution_research,
    step9_model_comparison,
)

STEPS = [
    ("Крок 1: Налаштування та завантаження", step1_setup.run_setup),
    ("Крок 2: Інференс на зображенні", step2_image_inference.run_image_inference),
    ("Крок 3: Геометричний аналіз IoU", step3_iou.run_iou_experiment),
    ("Крок 4: Алгоритми NMS", step4_nms.run_nms_experiment),
    ("Крок 5: Обробка відеопотоку", step5_video_processing.run_video_processing),
    ("Крок 6: Оцінка mAP на COCO128", step6_map_evaluation.run_map_evaluation),
    (
        "Крок 7: Дослідження Confidence Threshold",
        step7_confidence_research.run_confidence_research,
    ),
    (
        "Крок 8: Дослідження FPS vs Resolution",
        step8_fps_resolution_research.run_fps_resolution_research,
    ),
    (
        "Крок 9: Порівняння YOLOv8n та YOLOv8s",
        step9_model_comparison.run_model_comparison,
    ),
]


def main() -> int:
    """Головний цикл виконання лабораторної роботи № 5."""
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА № 5: ОБРОБКА МУЛЬТИМЕДІЙНИХ ДАНИХ")
    print("ТЕМА: ВИЯВЛЕННЯ ОБ'ЄКТІВ (OBJECT DETECTION) ЗА ДОПОМОГОЮ YOLO")
    print("=" * 80)

    total_start = time.perf_counter()

    for title, func in STEPS:
        print("\n" + "#" * 70)
        print(f"▶ {title.upper()}")
        print("#" * 70)
        step_start = time.perf_counter()
        func()
        step_elapsed = time.perf_counter() - step_start
        print(f"✔ Завершено за {step_elapsed:.1f}с")

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    print(f"УСІ ЕТАПИ ЛАБОРАТОРНОЇ РОБОТИ № 5 УСПІШНО ВИКОНАНО ЗА {total_elapsed:.1f}с!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
