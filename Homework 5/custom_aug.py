"""
Задание 2: Кастомные аугментации
"""

from extra_augs import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
import torch


def tensor_to_image(tensor):
    """
    Конвертирует тензор PyTorch в формат NumPy-изображения (H x W x C) для отображения.

    Args:
        tensor (torch.Tensor): Тензор изображения размером [C x H x W].

    Returns:
        np.ndarray: Изображение в формате NumPy с размерностью [H x W x C].
    """
    return tensor.permute(1, 2, 0).numpy()


# Инициализация пользовательских аугментаций с параметрами по умолчанию
custom_augmentations = [
    ("AddGaussianNoise", AddGaussianNoise(mean=0, std=0.1)),
    ("RandomErasingCustom", RandomErasingCustom(p=1.0, scale=(0.02, 0.2))),
    ("Solarize", Solarize(threshold=128)),
    ("ElasticTransform", ElasticTransform(p=1.0, alpha=30, sigma=5))
]

# Преобразование изображения в тензор
to_tensor = ToTensor()


# Визуализация всех пользовательских аугментаций для каждого класса
for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
    class_name = class_names[label]
    
    # Конвертируем исходное изображение в тензор
    tensor_img = to_tensor(img)
    
    # Создаём холст для отображения оригинального и аугментированных изображений
    plt.figure(figsize=(20, 10))
    
    # Отображаем оригинальное изображение
    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.title(f"Original\nClass: {class_name}")
    plt.axis('off')
    
    # Применяем каждую пользовательскую аугментацию и отображаем результат
    for j, (title, aug) in enumerate(custom_augmentations):
        plt.subplot(2, 4, j + 2)
        
        # Применяем аугментацию к скопированному тензору, чтобы не модифицировать оригинал
        augmented_tensor = aug(tensor_img.clone())
        
        # Конвертируем тензор обратно в изображение и обрезаем значения в диапазоне [0, 1]
        display_img = tensor_to_image(augmented_tensor)
        
        plt.imshow(np.clip(display_img, 0, 1))  # Обрезка значений для корректного отображения
        plt.title(f"{title}\nClass: {class_name}")
        plt.axis('off')
    
    # Сохраняем визуализацию в файл
    plt.tight_layout()
    plt.savefig(f"results/{class_name}_custom_augs.png", bbox_inches='tight')
    plt.close()
