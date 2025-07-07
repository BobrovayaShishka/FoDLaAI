"""
Задание 6. Дообучение предобученных моделей
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset  # Предполагается, что он реализован
import matplotlib.pyplot as plt
import numpy as np
import os
import time


# Создаем папку для сохранения результатов обучения
os.makedirs('results/training', exist_ok=True)

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# === Подготовка трансформаций изображений ===

# Трансформации для обучающей выборки (с аугментациями)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Трансформации для валидационной выборки (без аугментаций)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# === Загрузка датасетов ===

# Инициализируем обучающий и валидационный датасеты
train_dataset = CustomImageDataset('data/train', transform=train_transform)
val_dataset = CustomImageDataset('data/val', transform=val_transform)

# Создаем загрузчики данных
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False
)


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    """
    Обучает модель на заданном количестве эпох.
    
    Args:
        model (nn.Module): Модель PyTorch.
        criterion (nn.Module): Функция потерь.
        optimizer (torch.optim.Optimizer): Оптимизатор.
        scheduler (torch.optim.lr_scheduler): Шедулер изменения скорости обучения.
        num_epochs (int): Количество эпох обучения.

    Returns:
        tuple: (обученная модель, история метрик).
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_acc = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Обучение и валидация поочерёдно
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Обработка батчей
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Накопление статистики
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Сохраняем метрики
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())

                # Сохраняем лучшую модель
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()

    # Загружаем веса лучшей модели
    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_history(history, model_name):
    """
    Строит графики потерь и точности во время обучения.
    
    Args:
        history (dict): История обучения (потери и точность).
        model_name (str): Название модели для заголовков графиков.
    """
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/training/{model_name}_training_history.png')
    plt.close()


# === Эксперимент с разными моделями ===

# Список моделей для тестирования
models_to_train = {
    'resnet18': models.resnet18(weights='IMAGENET1K_V1'),
    'efficientnet_b0': models.efficientnet_b0(weights='IMAGENET1K_V1'),
    'mobilenet_v3_small': models.mobilenet_v3_small(weights='IMAGENET1K_V1')
}

# Получаем количество классов из датасета
num_classes = len(train_dataset.get_class_names())

# Обучаем каждую модель
for model_name, model in models_to_train.items():
    print(f"\n=== Training {model_name} ===")

    # Изменяем выходной слой под нашу задачу
    if model_name == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_small':
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    model = model.to(device)

    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Обучаем модель
    start_time = time.time()
    trained_model, history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs=10
    )
    training_time = time.time() - start_time

    # Сохраняем лучшую модель
    torch.save(trained_model.state_dict(), f'results/training/{model_name}_best.pth')

    # Визуализируем обучение
    plot_training_history(history, model_name)

    # Сохраняем метрики в файл
    best_val_acc = max(history['val_acc'])
    with open('results/training/summary.txt', 'a') as f:
        f.write(f"{model_name}: Best Val Acc = {best_val_acc:.4f}, Training Time = {training_time:.2f}s\n")

    print(f"Training completed for {model_name} in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


def compare_models():
    """
    Сравнивает обученные модели по точности и времени обучения.
    Строит графики сравнения.
    """
    model_names = []
    val_accs = []
    train_times = []

    # Читаем результаты из файла
    with open('results/training/summary.txt', 'r') as f:
        for line in f:
            parts = line.split(':')
            model_name = parts[0]
            metrics = parts[1].split(',')
            acc = float(metrics[0].split('=')[1].strip())
            time = float(metrics[1].split('=')[1].strip().replace('s', ''))

            model_names.append(model_name)
            val_accs.append(acc)
            train_times.append(time)

    # График точности
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, val_accs, color='skyblue')
    plt.title('Model Comparison: Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)
    plt.savefig('results/training/model_comparison_acc.png')
    plt.close()

    # График времени обучения
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, train_times, color='lightgreen')
    plt.title('Model Comparison: Training Time')
    plt.ylabel('Time (seconds)')
    plt.savefig('results/training/model_comparison_time.png')
    plt.close()


# Вызываем функцию сравнения
compare_models()
