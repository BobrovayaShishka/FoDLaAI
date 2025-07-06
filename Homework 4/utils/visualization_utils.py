import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def ensure_dir(directory):
    """Создает директорию если не существует"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_training_history(history, title, save_dir='plots'):
    """Визуализирует историю обучения с сохранением в файл"""
    ensure_dir(save_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title, save_dir='plots'):
    """Визуализирует матрицу ошибок"""
    ensure_dir(save_dir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_activations(activations, title, save_dir='plots'):
    """Визуализирует активации сверточных слоев"""
    ensure_dir(save_dir)
    
    # Проверяем и корректируем форму активаций
    if activations.ndim == 4:
        # Если активации имеют форму [batch, channels, H, W]
        activations = activations[0]  # Берем первый образец батча
    elif activations.ndim == 1:
        # Если активации имеют форму [channels] - преобразуем в [channels, 1, 1]
        activations = activations[:, np.newaxis, np.newaxis]
    
    # Получаем количество каналов
    num_channels = activations.shape[0]
    
    # Рассчитываем размер сетки для отображения
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    if grid_size * (grid_size - 1) >= num_channels:
        rows = grid_size - 1
    else:
        rows = grid_size
    
    plt.figure(figsize=(12, 8))
    for i in range(min(16, num_channels)):
        plt.subplot(4, 4, i+1)
        
        # Для 1D активаций отображаем как гистограмму
        if activations[i].ndim == 1 or activations[i].size == 1:
            plt.hist(activations[i].flatten(), bins=20, alpha=0.7)
            plt.title(f'Chan {i+1} Dist')
        # Для 2D+ активаций отображаем как изображение
        else:
            # Берем среднее по каналам если нужно
            if activations[i].ndim > 2:
                img = activations[i].mean(axis=0)
            else:
                img = activations[i]
                
            plt.imshow(img, cmap='viridis')
            plt.title(f'Chan {i+1}')
        
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_gradients(grad_means, layer_names, title, save_dir='plots'):
    """Визуализирует градиенты по слоям"""
    ensure_dir(save_dir)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(grad_means)), grad_means)
    plt.xticks(range(len(grad_means)), layer_names, rotation=45)
    plt.yscale('log')
    plt.title(title)
    plt.ylabel('Mean Absolute Gradient (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_model_comparison(histories, labels, metric='test_accs', 
                         title='Model Comparison', save_dir='plots'):
    """Сравнивает модели по метрике"""
    ensure_dir(save_dir)
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history[metric], label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy' if 'acc' in metric else 'Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_receptive_field(rf_sizes, title, save_dir='plots'):
    """Визуализирует размер рецептивных полей"""
    ensure_dir(save_dir)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rf_sizes)+1), rf_sizes, marker='o')
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Receptive Field Size (pixels)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()
