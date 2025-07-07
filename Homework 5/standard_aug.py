"""
Задание 1: Стандартные аугментации torchvision
"""

import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

# Импортируем пользовательские классы и функции
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, CutOut, Solarize


def set_seed(seed=42):
    """
    Фиксирует случайные числа для воспроизводимости результатов.

    Args:
        seed (int): Значение начального числа для генератора случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Создаем папку 'results' для сохранения визуализаций
os.makedirs('results', exist_ok=True)

# Загружаем кастомный датасет из указанной директории
root = 'data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))

# Выбираем по одному изображению из первых 5 классов для дальнейшей обработки
class_names = dataset.get_class_names()[:6]  # Берём только первые 5 классов
sample_images = []
sample_labels = []

for class_name in class_names:
    # Находим индекс первого изображения в текущем классе
    idx = next(i for i, label in enumerate(dataset.labels)
               if dataset.classes[label] == class_name)
    img, label = dataset[idx]
    sample_images.append(img)  # Добавляем изображение в список
    sample_labels.append(label)  # Добавляем соответствующую метку

# Определяем список стандартных аугментаций с описаниями
standard_augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0)),
]

# Комбинированная аугментация — применяется несколько преобразований последовательно
combined_standard = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(200, padding=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.2),
])


def visualize_augmentations(original_img, augmented_list, titles, filename):
    """
    Визуализирует оригинальное изображение и его аугментированные версии.

    Args:
        original_img (PIL.Image): Оригинальное изображение.
        augmented_list (list): Список аугментированных изображений.
        titles (list): Список заголовков для отображения на графиках.
        filename (str): Путь для сохранения результата визуализации.
    """
    plt.figure(figsize=(15, 5))

    # Отображаем оригинальное изображение
    plt.subplot(1, len(augmented_list) + 1, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis('off')

    # Отображаем все аугментированные изображения
    for i, (img, title) in enumerate(zip(augmented_list, titles)):
        plt.subplot(1, len(augmented_list) + 1, i + 2)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# Применяем аугментации к каждому из выбранных изображений
for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
    class_name = class_names[label]
    augmented_imgs = []  # Хранилище аугментированных изображений
    titles = []  # Хранилище названий аугментаций

    # Применяем каждую аугментацию из списка стандартных отдельно
    for name, aug in standard_augs:
        augmented = aug(img)
        augmented_imgs.append(augmented)
        titles.append(name)

    # Применяем комбинированную аугментацию
    combined_img = combined_standard(img)
    augmented_imgs.append(combined_img)
    titles.append("Combined")

    # Сохраняем результат визуализации в файл
    visualize_augmentations(
        img,
        augmented_imgs,
        titles,
        f"results/{class_name}_standard_augs.png"
    )
