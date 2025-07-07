"""
Задание 3. Анализ датасета
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict


def analyze_dataset(root_dir):
    """
    Анализирует структуру датасета: считает количество изображений в каждом классе,
    а также собирает информацию о ширине и высоте изображений.

    Args:
        root_dir (str): Путь к директории с данными, где каждый подкаталог — это класс.

    Returns:
        tuple: Кортеж из двух элементов:
            - class_counts (defaultdict): Словарь с количеством изображений на класс.
            - size_stats (dict): Статистика по размерам изображений:
                - min_width, max_width, mean_width
                - min_height, max_height, mean_height
                - total_images
                - all_widths, all_heights (списки всех значений)
    """
    # Инициализируем словарь для подсчёта числа изображений по классам
    class_counts = defaultdict(int)

    # Списки для хранения ширины и высоты всех изображений
    widths = []
    heights = []

    # Получаем список классов (папок) в директории
    classes = sorted([d for d in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, d))])

    # Проходим по каждому классу
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        # Подсчитываем количество изображений в классе
        class_counts[class_name] = len(image_files)

        # Собираем информацию о размерах изображений
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)

    # Преобразуем списки в массивы NumPy для вычисления статистики
    widths = np.array(widths)
    heights = np.array(heights)

    # Рассчитываем статистику по размерам
    size_stats = {
        'min_width': widths.min(),
        'max_width': widths.max(),
        'mean_width': widths.mean(),
        'min_height': heights.min(),
        'max_height': heights.max(),
        'mean_height': heights.mean(),
        'total_images': len(widths),
        'all_widths': widths,
        'all_heights': heights
    }

    return class_counts, size_stats


# ==== Основная часть программы ====
if __name__ == "__main__":
    # Директория с валидационной выборкой
    root_dir = 'data/val'

    # Выполняем анализ датасета
    class_counts, size_stats = analyze_dataset(root_dir)

    # Создаём папку для сохранения результатов
    os.makedirs('results', exist_ok=True)

    # Сохраняем текстовые результаты в файл
    with open('results/dataset_analysis.txt', 'w') as f:
        f.write("=== Размеры изображений ===\n")
        f.write(f"Минимальная ширина: {size_stats['min_width']}\n")
        f.write(f"Максимальная ширина: {size_stats['max_width']}\n")
        f.write(f"Средняя ширина: {size_stats['mean_width']:.2f}\n")
        f.write(f"Минимальная высота: {size_stats['min_height']}\n")
        f.write(f"Максимальная высота: {size_stats['max_height']}\n")
        f.write(f"Средняя высота: {size_stats['mean_height']:.2f}\n")
        f.write(f"Всего изображений: {size_stats['total_images']}\n\n")

        f.write("=== Распределение по классам ===\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count} изображений\n")

    # ==== Визуализация результатов ====

    # Создаём фигуру для отображения нескольких графиков
    plt.figure(figsize=(15, 10))

    # Гистограмма распределения ширин
    plt.subplot(2, 2, 1)
    plt.hist(size_stats['all_widths'], bins=50, alpha=0.7, color='blue')
    plt.title('Распределение ширин')
    plt.xlabel('Ширина (px)')
    plt.ylabel('Количество')

    # Гистограмма распределения высот
    plt.subplot(2, 2, 2)
    plt.hist(size_stats['all_heights'], bins=50, alpha=0.7, color='green')
    plt.title('Распределение высот')
    plt.xlabel('Высота (px)')
    plt.ylabel('Количество')

    # Распределение соотношения сторон (ширина / высота)
    plt.subplot(2, 2, 3)
    aspect_ratios = size_stats['all_widths'] / size_stats['all_heights']
    plt.hist(aspect_ratios, bins=50, alpha=0.7, color='purple')
    plt.title('Распределение соотношений сторон (w/h)')
    plt.xlabel('Соотношение сторон')
    plt.ylabel('Количество')

    # График распределения изображений по классам
    plt.subplot(2, 2, 4)
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    plt.bar(class_names, counts, alpha=0.7)
    plt.title('Количество изображений по классам')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    plt.xticks(rotation=45, ha='right')  # Поворачиваем подписи классов для лучшего отображения

    # Корректируем расположение графиков
    plt.tight_layout()

    # Сохраняем изображение с графиками
    plt.savefig('results/dataset_distribution.png', bbox_inches='tight')

    # Закрываем рисунок
    plt.close()
