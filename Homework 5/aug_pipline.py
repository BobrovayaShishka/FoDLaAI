"""
задание 4. Pipeline аугментаций
"""

import time
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class AugmentationPipeline:
    """
    Класс, представляющий гибкий пайплайн аугментаций.

    Позволяет добавлять, удалять и применять аугментации к изображениям.
    """

    def __init__(self):
        """Инициализирует пустой словарь аугментаций и пустой пайплайн."""
        self.augmentations = {}
        self.pipeline = transforms.Compose([])

    def add_augmentation(self, name, augmentation):
        """
        Добавляет новую аугментацию в пайплайн.

        Args:
            name (str): Имя аугментации (для идентификации).
            augmentation (transforms.Transform): Объект аугментации torchvision.
        """
        self.augmentations[name] = augmentation
        self._update_pipeline()

    def remove_augmentation(self, name):
        """
        Удаляет аугментацию по имени, если она существует.

        Args:
            name (str): Имя аугментации для удаления.
        """
        if name in self.augmentations:
            del self.augmentations[name]
            self._update_pipeline()

    def apply(self, image):
        """
        Применяет текущий пайплайн аугментаций к изображению.

        Args:
            image (PIL.Image or torch.Tensor): Входное изображение.

        Returns:
            PIL.Image or torch.Tensor: Аугментированное изображение.
        """
        return self.pipeline(image)

    def get_augmentations(self):
        """
        Возвращает список имён всех активных аугментаций.

        Returns:
            list: Список названий аугментаций.
        """
        return list(self.augmentations.keys())

    def _update_pipeline(self):
        """Обновляет внутренний Compose-пайплайн на основе текущих аугментаций."""
        self.pipeline = transforms.Compose(list(self.augmentations.values()))


def create_light_config():
    """
    Создаёт легкие аугментации — минимальные искажения.

    Returns:
        list: Список кортежей (имя, объект аугментации).
    """
    return [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.3)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.1, contrast=0.1))
    ]


def create_medium_config():
    """
    Создаёт умеренные аугментации — баланс между разнообразием и стабильностью.

    Returns:
        list: Список кортежей (имя, объект аугментации).
    """
    return [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.5)),
        ("RandomRotation", transforms.RandomRotation(15)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)),
        ("RandomCrop", transforms.RandomCrop(64, padding=20))
    ]


def create_heavy_config():
    """
    Создаёт агрессивные аугментации — значительные изменения изображений.

    Returns:
        list: Список кортежей (имя, объект аугментации).
    """
    return [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.7)),
        ("RandomRotation", transforms.RandomRotation(30)),
        ("ColorJitter", transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)),
        ("RandomBlur", RandomBlur(p=0.3, max_radius=5)),
        ("AddGaussianNoise", transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.2)
        ]))
    ]


def apply_and_save_pipeline(config, name, sample_images):
    """
    Применяет указанную конфигурацию аугментаций к выборке изображений и сохраняет результаты.

    Args:
        config (list): Список кортежей (имя, объект аугментации).
        name (str): Название конфигурации (например, 'light', 'medium', 'heavy').
        sample_images (list): Список изображений (PIL.Image или torch.Tensor).
    """
    # Инициализируем пайплайн и добавляем аугментации
    pipeline = AugmentationPipeline()
    for aug_name, aug in config:
        pipeline.add_augmentation(aug_name, aug)

    # Применяем пайплайн к каждому изображению и сохраняем результаты
    for i, img in enumerate(sample_images):
        augmented_img = pipeline.apply(img)

        # Конвертируем аугментированное изображение в NumPy массив [0, 1]
        if isinstance(augmented_img, torch.Tensor):
            img_np = augmented_img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(augmented_img) / 255.0  # PIL -> нормализованный NumPy
        img_np = np.clip(img_np, 0, 1)  # Ограничиваем значения

        # Конвертируем оригинальное изображение аналогично
        if isinstance(img, torch.Tensor):
            orig_np = img.permute(1, 2, 0).numpy()
        else:
            orig_np = np.array(img) / 255.0
        orig_np = np.clip(orig_np, 0, 1)

        # Визуализация: оригинал и аугментация
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(orig_np)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        plt.title(f"{name} Augmentation")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"results/{name}_aug_{i}.png", bbox_inches='tight')
        plt.close()


# === Основной блок запуска ===
if __name__ == "__main__":
    # Выбираем 3 случайных изображения из датасета
    sample_indices = random.sample(range(len(dataset)), 3)
    sample_images = [dataset[i][0] for i in sample_indices]

    # Создаём три конфигурации аугментаций
    configs = {
        "light": create_light_config(),
        "medium": create_medium_config(),
        "heavy": create_heavy_config()
    }

    # Применяем и сохраняем результаты
    for name, config in configs.items():
        apply_and_save_pipeline(config, name, sample_images)
