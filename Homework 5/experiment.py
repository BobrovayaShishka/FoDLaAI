"""
Задание 5.  Эксперимент с размерами
"""

import time
import psutil
import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from my_datasets import CustomImageDataset
from augmentation_pipeline import AugmentationPipeline, create_medium_config


def measure_performance(root_dir, sizes, num_images=100):
    """
    Измеряет производительность загрузки и обработки изображений разных размеров.

    Args:
        root_dir (str): Путь к директории с данными.
        sizes (list): Список размеров изображений для тестирования.
        num_images (int): Количество изображений для анализа на каждом размере.

    Returns:
        dict: Словарь с результатами:
            - 'load_times': время загрузки изображений
            - 'aug_times': время применения аугментаций
            - 'memory_usages': использование памяти
    """
    # Словарь для хранения результатов
    results = {
        'load_times': [],
        'aug_times': [],
        'memory_usages': []
    }

    # Проходим по каждому заданному размеру изображения
    for size in sizes:
        # Создаём датасет с текущим целевым размером
        dataset = CustomImageDataset(
            root_dir,
            transform=None,
            target_size=(size, size)
        )

        # Выбираем случайные индексы для анализа
        indices = random.sample(range(len(dataset)), num_images)

        # --- Измерение времени загрузки ---
        start_time = time.time()
        images = [dataset[i][0] for i in indices]
        load_time = time.time() - start_time

        # --- Подготовка пайплайна аугментаций ---
        pipeline = AugmentationPipeline()
        for _, aug in create_medium_config():
            pipeline.add_augmentation("temp", aug)

        # --- Измерение времени аугментации ---
        start_time = time.time()
        for img in images:
            _ = pipeline.apply(img)
        aug_time = time.time() - start_time

        # --- Измерение использования памяти ---
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 ** 2  # Переводим в МБ

        # Очищаем кэш GPU, если доступен
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Преобразуем изображения в тензоры для оценки потребления памяти
        tensors = [transforms.ToTensor()(img) for img in images]
        tensor_mem = sum(t.nelement() * t.element_size() for t in tensors) / 1024 ** 2

        # Замеряем память после преобразования
        mem_after = process.memory_info().rss / 1024 ** 2
        mem_usage = mem_after - mem_before + tensor_mem  # Учитываем объём тензоров

        # Сохраняем результаты
        results['load_times'].append(load_time)
        results['aug_times'].append(aug_time)
        results['memory_usages'].append(mem_usage)

        # Вывод промежуточных результатов
        print(f"Size: {size}x{size} | "
              f"Load: {load_time:.2f}s | "
              f"Aug: {aug_time:.2f}s | "
              f"Memory: {mem_usage:.2f}MB")

    return results


# === Основной блок запуска ===
if __name__ == "__main__":
    # Параметры эксперимента
    sizes = [64, 128, 224, 512]  # Тестируемые размеры изображений
    num_images = 100           # Число изображений для анализа
    data_path = 'data/test'    # Путь к тестовым данным

    # Проведение замеров
    results = measure_performance(data_path, sizes, num_images=num_images)

    # Создаём папку для сохранения графиков
    os.makedirs('results', exist_ok=True)

    # === Визуализация результатов ===
    plt.figure(figsize=(15, 10))

    # --- Время загрузки ---
    plt.subplot(2, 2, 1)
    plt.plot(sizes, results['load_times'], 'o-')
    plt.title('Время загрузки 100 изображений')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время (сек)')
    plt.grid(True)

    # --- Время аугментации ---
    plt.subplot(2, 2, 2)
    plt.plot(sizes, results['aug_times'], 'o-', color='orange')
    plt.title('Время аугментации 100 изображений')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время (сек)')
    plt.grid(True)

    # --- Использование памяти ---
    plt.subplot(2, 2, 3)
    plt.plot(sizes, results['memory_usages'], 'o-', color='green')
    plt.title('Использование памяти')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Память (MB)')
    plt.grid(True)

    # --- Общее время обработки ---
    plt.subplot(2, 2, 4)
    total_time = np.array(results['load_times']) + np.array(results['aug_times'])
    plt.plot(sizes, total_time, 'o-', color='red')
    plt.title('Общее время обработки')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время (сек)')
    plt.grid(True)

    # Корректировка отступов и сохранение результата
    plt.tight_layout()
    plt.savefig('results/size_performance.png', bbox_inches='tight')
    plt.close()
