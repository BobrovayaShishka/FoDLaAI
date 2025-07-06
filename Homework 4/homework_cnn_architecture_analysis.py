import time
import logging
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import cnn_models
from datasets import get_cifar_loaders
from utils.training_utils import train_model, count_parameters
from utils.visualization_utils import (
    plot_training_history, 
    plot_activations,
    plot_gradients,
    plot_receptive_field,
    plot_model_comparison,
    ensure_dir
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("architecture_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def get_gradients(model, data_loader, criterion, device):
    """Собирает градиенты по слоям модели"""
    model.train()
    gradients = {}
    
    hooks = []
    def hook_fn(module, grad_input, grad_output, name):
        if grad_output[0] is not None:
            gradients[name] = grad_output[0].abs().mean().item()
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_full_backward_hook(
                lambda module, grad_input, grad_output, name=name: 
                hook_fn(module, grad_input, grad_output, name)
            )
            hooks.append(hook)
    
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    for hook in hooks:
        hook.remove()
        
    return gradients

def kernel_size_experiment():
    """Эксперимент с размером ядра свертки"""
    logger.info("\n" + "="*50)
    logger.info("Starting Kernel Size Experiment")
    logger.info("="*50)
    save_dir = "results/architecture_analysis/kernel_size"
    ensure_dir(save_dir)
    
    # Гиперпараметры
    batch_size = 128
    epochs = 15
    lr = 0.001

    # Загрузка данных
    train_loader, test_loader = get_cifar_loaders(batch_size)

    # Конфигурации моделей
    configs = [
        {'name': '3x3', 'kernels': [3, 3, 3]},
        {'name': '5x5', 'kernels': [5, 5, 5]},
        {'name': '7x7', 'kernels': [7, 7, 7]},
        {'name': '1x1+3x3', 'kernels': [1, 3, 3]},
    ]

    results = {}
    for cfg in configs:
        name = cfg['name']
        model = cnn_models.KernelSizeCNN(cfg['kernels']).to(device)
        params = count_parameters(model)
        logger.info(f"\nTraining {name} model... Parameters: {params:,}")

        # Визуализация рецептивных полей
        rf_sizes = []
        current_rf = 1
        stride_product = 1
        for kernel in cfg['kernels']:
            current_rf = current_rf + (kernel - 1) * stride_product
            rf_sizes.append(current_rf)
            stride_product *= 2  # После каждого пулинга
        plot_receptive_field(
            rf_sizes, 
            f"{name} Kernel Receptive Field",
            save_dir
        )

        # Тренировка
        start_time = time.time()
        history = train_model(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        train_time = time.time() - start_time

        # Визуализация активаций первого слоя
        data, _ = next(iter(train_loader))
        data = data[:1].to(device)
        
        # Регистрируем хук для получения активаций
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        
        hook = model.conv1.register_forward_hook(hook_fn)
        
        # Прямой проход
        with torch.no_grad():
            model(data)
        
        # Убираем хук
        hook.remove()
        
        # Получаем активации
        conv1_activations = activations[0][0]  # [1, C, H, W] -> [C, H, W]
        
        # Визуализируем
        plot_activations(
            conv1_activations.numpy(), 
            f"{name} First Layer Activations",
            save_dir
        )

        # Сохранение результатов
        results[name] = {
            'history': history,
            'train_time': train_time,
            'params': params,
            'test_acc': history['test_accs'][-1]
        }

        # Визуализация истории обучения
        plot_training_history(
            history, 
            f"{name} Kernel - Training History",
            save_dir
        )

    # Сравнение результатов
    logger.info("\nKernel Size Results:")
    for name, res in results.items():
        logger.info(
            f"{name}: "
            f"Test Acc = {res['test_acc']:.4f}, "
            f"Train Time = {res['train_time']:.2f}s, "
            f"Params = {res['params']:,}"
        )

    # Визуализация сравнения
    histories = [res['history'] for res in results.values()]
    labels = list(results.keys())
    plot_model_comparison(
        histories, labels, 
        title="Kernel Size Comparison (Test Accuracy)",
        save_dir=save_dir
    )

def depth_experiment():
    """Эксперимент с глубиной сети"""
    logger.info("\n" + "="*50)
    logger.info("Starting Network Depth Experiment")
    logger.info("="*50)
    save_dir = "results/architecture_analysis/depth"
    ensure_dir(save_dir)
    
    # Гиперпараметры
    batch_size = 128
    epochs = 20
    lr = 0.001

    # Загрузка данных
    train_loader, test_loader = get_cifar_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    # Конфигурации моделей
    configs = [
        {'name': 'Shallow (2 blocks)', 'depth': 2, 'residual': False},
        {'name': 'Medium (4 blocks)', 'depth': 4, 'residual': False},
        {'name': 'Deep (6 blocks)', 'depth': 6, 'residual': False},
        {'name': 'Residual (6 blocks)', 'depth': 6, 'residual': True},
    ]

    results = {}
    for cfg in configs:
        name = cfg['name']
        model = cnn_models.DepthCNN(
            num_blocks=cfg['depth'],
            use_residual=cfg['residual']
        ).to(device)
        params = count_parameters(model)
        logger.info(f"\nTraining {name} model... Parameters: {params:,}")

        # Тренировка
        start_time = time.time()
        history = train_model(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        train_time = time.time() - start_time

        # Анализ градиентов
        gradients = get_gradients(model, train_loader, criterion, device)
        layer_names = list(gradients.keys())
        grad_means = list(gradients.values())
        plot_gradients(
            grad_means, layer_names,
            f"{name} Gradient Analysis",
            save_dir
        )

        # Визуализация активаций
        data, _ = next(iter(train_loader))
        data = data[:1].to(device)
        
        # Регистрируем хук для первого сверточного слоя
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        
        hook = model.conv1.register_forward_hook(hook_fn)
        
        # Прямой проход
        with torch.no_grad():
            model(data)
        
        # Убираем хук
        hook.remove()
        
        # Получаем активации
        conv1_activations = activations[0][0]  # [1, C, H, W] -> [C, H, W]
        
        # Визуализируем
        plot_activations(
            conv1_activations.numpy(), 
            f"{name} First Layer Activations",
            save_dir
        )

        # Confusion matrix
        all_preds = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Сохранение результатов
        results[name] = {
            'history': history,
            'train_time': train_time,
            'params': params,
            'test_acc': history['test_accs'][-1],
            'grad_means': grad_means
        }

        # Визуализация истории обучения
        plot_training_history(
            history, 
            f"{name} - Training History",
            save_dir
        )

    # Сравнение результатов
    logger.info("\nDepth Experiment Results:")
    for name, res in results.items():
        logger.info(
            f"{name}: "
            f"Test Acc = {res['test_acc']:.4f}, "
            f"Train Time = {res['train_time']:.2f}s, "
            f"Params = {res['params']:,}"
        )

    # Визуализация сравнения точности
    histories = [res['history'] for res in results.values()]
    labels = list(results.keys())
    plot_model_comparison(
        histories, labels, 
        title="Network Depth Comparison (Test Accuracy)",
        save_dir=save_dir
    )

    # Анализ градиентов (средние значения)
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        if 'grad_means' in res:
            plt.plot(res['grad_means'], label=name, marker='o')

    plt.title("Gradient Magnitude Comparison")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Absolute Gradient (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient_comparison.png"))
    plt.close()

if __name__ == "__main__":
    kernel_size_experiment()
    depth_experiment()
