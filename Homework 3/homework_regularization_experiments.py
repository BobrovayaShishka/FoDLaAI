import os
import json
import time
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataset_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import generate_config, create_model, count_parameters
from utils.visualization_utils import plot_training_history, plot_comparison_bar, plot_weight_distribution

# Настройка путей для сохранения результатов
RESULTS_DIR = "results/regularization_experiments"  # Основная директория для результатов
ADAPTIVE_DIR = os.path.join(RESULTS_DIR, "adaptive")  # Для адаптивных методов
WEIGHT_DIST_DIR = os.path.join(RESULTS_DIR, "weight_distributions")  # Для распределений весов
PLOTS_DIR = "plots/regularization_experiments"  # Для графиков
os.makedirs(RESULTS_DIR, exist_ok=True)  # Создаем директории, если они не существуют
os.makedirs(ADAPTIVE_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIST_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Основные параметры эксперимента
BATCH_SIZE = 128  # Размер батча
EPOCHS = 30  # Количество эпох обучения
LR = 0.001  # Скорость обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Используем GPU, если доступен
BASE_WIDTHS = [1024, 512, 256]  # Базовая архитектура сети

def setup_logger(dataset_name):
    """Настройка логгера для записи хода эксперимента"""
    logger = logging.getLogger(f'reg_experiment_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # Файловый обработчик для записи логов
    file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, f'{dataset_name}_experiment.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def run_epoch(model, data_loader, criterion, optimizer=None, is_test=False):
    """Выполнение одной эпохи обучения или тестирования"""
    if is_test:
        model.eval()  # Переводим модель в режим тестирования
    else:
        model.train()  # Режим обучения
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Итерация по батчам с прогресс-баром
    for data, target in tqdm(data_loader, desc=f"{'Test' if is_test else 'Train'}"):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        if not is_test:
            optimizer.zero_grad()  # Обнуляем градиенты
        
        output = model(data)  # Прямой проход
        loss = criterion(output, target)  # Вычисление потерь
        
        if not is_test:
            loss.backward()  # Обратное распространение
            optimizer.step()  # Обновление весов
        
        # Сбор статистики
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    # Средние значения по эпохе
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, config, exp_name, dataset_name, epochs=EPOCHS, lr=LR):
    """Полный цикл обучения модели"""
    criterion = nn.CrossEntropyLoss()  # Функция потерь
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=config.get('weight_decay', 0.0)  # L2-регуляризация
    )
    
    # Словарь для хранения истории обучения
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': [],
        'weight_stats': []  # Статистика весов
    }
    
    logger = logging.getLogger(f'reg_experiment_{dataset_name}')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Адаптивный dropout (если используется)
        if config.get('adaptive_dropout', False):
            current_dropout = config['dropout_rate'] * (epoch / epochs)
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = current_dropout
            logger.info(f"Epoch {epoch+1}: Adaptive dropout rate = {current_dropout:.3f}")
        
        # Обучение на тренировочных данных
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        
        # Тестирование на валидационных данных
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, is_test=True
        )
        
        # Сохранение метрик
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        # Сбор статистики по весам
        weight_stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' not in name:  # Исключаем BatchNorm
                weight_stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'abs_mean': param.data.abs().mean().item()
                }
        history['weight_stats'].append(weight_stats)
        
        # Визуализация распределения весов каждые 5 эпох
        if (epoch % 5 == 0) or (epoch == epochs - 1):
            weight_plot_dir = os.path.join(WEIGHT_DIST_DIR, exp_name)
            os.makedirs(weight_plot_dir, exist_ok=True)
            plot_weight_distribution(
                model, epoch+1, 
                f"{exp_name} - Epoch {epoch+1}",
                os.path.join(weight_plot_dir, f"epoch_{epoch+1}.png")
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                   f"Time: {epoch_time:.2f} sec")
    
    # Сохранение статистики весов
    weight_stats_path = os.path.join(WEIGHT_DIST_DIR, f"{exp_name}_weight_stats.json")
    with open(weight_stats_path, 'w') as f:
        json.dump(history['weight_stats'], f, indent=2)
    
    return history

def regularization_experiment(dataset_name, test_mode=False):
    """Основной эксперимент по сравнению методов регуляризации"""
    logger = setup_logger(dataset_name)
    logger.info(f"Starting regularization experiment for {dataset_name}")
    
    if test_mode:
        epochs = 2  # Уменьшенное количество эпох для тестового режима
        logger.info("Test mode activated")
    else:
        epochs = EPOCHS
    
    # Загрузка данных
    loader_fn = get_mnist_loaders if dataset_name == "mnist" else get_cifar_loaders
    train_loader, test_loader = loader_fn(batch_size=BATCH_SIZE)
    
    # Определение размеров входных данных
    sample, _ = next(iter(train_loader))
    input_size = sample.view(sample.size(0), -1).shape[1]
    num_classes = len(train_loader.dataset.dataset.classes)
    
    results = {}  # Для хранения результатов всех экспериментов
    
    # Часть 1: Сравнение методов регуляризации
    experiments = [
        {"name": "no_reg", "config": generate_config(BASE_WIDTHS)},  # Без регуляризации
        {"name": "dropout_0.1", "config": generate_config(BASE_WIDTHS, dropout_rate=0.1)},
        {"name": "dropout_0.3", "config": generate_config(BASE_WIDTHS, dropout_rate=0.3)},
        {"name": "dropout_0.5", "config": generate_config(BASE_WIDTHS, dropout_rate=0.5)},
        {"name": "batchnorm", "config": generate_config(BASE_WIDTHS, use_batchnorm=True)},
        {"name": "both", "config": generate_config(BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True)},
        {"name": "l2_1e-4", "config": generate_config(BASE_WIDTHS, weight_decay=1e-4)},  # L2-регуляризация
        {"name": "l2_1e-3", "config": generate_config(BASE_WIDTHS, weight_decay=1e-3)},
        {"name": "l2_1e-2", "config": generate_config(BASE_WIDTHS, weight_decay=1e-2)},
    ]
    
    for exp in experiments:
        logger.info(f"\n=== Experiment: {exp['name']} ===")
        
        # Создание модели
        model = create_model(exp['config'], input_size, num_classes).to(DEVICE)
        
        # Настройка BatchNorm
        if exp['config'].get('use_batchnorm', False):
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.momentum = exp['config'].get('bn_momentum', 0.1)
        
        param_count = count_parameters(model)
        logger.info(f"Model parameters: {param_count:,}")
        
        # Обучение модели
        history = train_model(
            model, train_loader, test_loader, 
            exp['config'], exp['name'], dataset_name, epochs
        )
        
        # Сохранение результатов
        results[exp['name']] = {
            'history': history,
            'config': exp['config'],
            'params': param_count
        }
        
        # Визуализация истории обучения
        plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_{exp['name']}.png")
        plot_training_history(
            history, 
            f"{dataset_name} - {exp['name']}", 
            plot_path
        )
    
    # Часть 2: Адаптивные методы регуляризации
    if not test_mode:
        adaptive_experiments = [
            {"name": "adaptive_dropout", "config": generate_config(
                BASE_WIDTHS, dropout_rate=0.5, adaptive_dropout=True)},
            {"name": "bn_momentum_0.5", "config": generate_config(
                BASE_WIDTHS, use_batchnorm=True, bn_momentum=0.5)},
            {"name": "bn_momentum_0.9", "config": generate_config(
                BASE_WIDTHS, use_batchnorm=True, bn_momentum=0.9)},
            {"name": "combined", "config": generate_config(
                BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True, weight_decay=1e-4)},
            {"name": "layer_specific", "config": generate_config(
                BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True, layer_specific=True)},
        ]
        
        for exp in adaptive_experiments:
            logger.info(f"\n=== Adaptive Experiment: {exp['name']} ===")
            
            model = create_model(exp['config'], input_size, num_classes).to(DEVICE)
            
            if exp['config'].get('use_batchnorm', False):
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.momentum = exp['config'].get('bn_momentum', 0.1)
            
            param_count = count_parameters(model)
            logger.info(f"Model parameters: {param_count:,}")
            
            history = train_model(
                model, train_loader, test_loader, 
                exp['config'], exp['name'], dataset_name, epochs
            )
            
            results[exp['name']] = {
                'history': history,
                'config': exp['config'],
                'params': param_count
            }
            
            # Сохранение адаптивных результатов отдельно
            adaptive_path = os.path.join(ADAPTIVE_DIR, f"{dataset_name}_{exp['name']}.json")
            with open(adaptive_path, 'w') as f:
                json.dump({
                    'history': history,
                    'config': exp['config'],
                    'params': param_count
                }, f, indent=2)
            
            plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_{exp['name']}.png")
            plot_training_history(
                history, 
                f"{dataset_name} - {exp['name']}", 
                plot_path
            )
    
    # Анализ и сохранение результатов
    analyze_results(results, dataset_name, logger)
    
    results_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results

def analyze_results(results, dataset_name, logger):
    """Анализ и визуализация результатов экспериментов"""
    names = list(results.keys())
    test_accs = [results[name]['history']['test_acc'][-1] for name in names]
    train_times = [sum(results[name]['history']['epoch_time']) for name in names]
    
    # Сравнение точности разных методов
    acc_path = os.path.join(PLOTS_DIR, f"{dataset_name}_comparison.png")
    plot_comparison_bar(
        test_accs, names, 
        f"{dataset_name}: Regularization Techniques Comparison",
        "Accuracy",
        acc_path
    )
    
    # Логирование сводки результатов
    logger.info("\n" + "="*50)
    logger.info(f"{dataset_name.upper()} - Results Summary:")
    for name in names:
        r = results[name]
        logger.info(f"{name}: Test Acc: {r['history']['test_acc'][-1]:.4f}, "
                   f"Time: {sum(r['history']['epoch_time']):.1f} sec, "
                   f"Params: {r['params']:,}")

def main():
    """Точка входа в программу"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True)
    parser.add_argument('--test', action='store_true', help="Test mode")
    args = parser.parse_args()
    
    regularization_experiment(args.dataset, test_mode=args.test)

if __name__ == "__main__":
    main()
