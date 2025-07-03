import os
import json
import time
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.dataset_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import generate_width_config, create_model, count_parameters
from utils.visualization_utils import plot_training_history, plot_comparison_bar, plot_heatmap

# ==================== КОНФИГУРАЦИЯ ПУТЕЙ И ПАРАМЕТРОВ ====================

# Пути для сохранения результатов
RESULTS_DIR = "results/width_experiments"          # Основная папка с результатами
GRID_SEARCH_DIR = os.path.join(RESULTS_DIR, "grid_search")  # Для результатов grid search
PLOTS_DIR = "plots/width_experiments"             # Для графиков

# Создаем необходимые директории
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRID_SEARCH_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Гиперпараметры обучения
BATCH_SIZE = 128     # Размер батча
EPOCHS = 20          # Количество эпох для основных экспериментов
LR = 0.001           # Скорость обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Устройство для вычислений

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def setup_logger(dataset_name):
    """
    Настройка логгера для записи хода экспериментов.
    
    Args:
        dataset_name (str): Имя датасета ('mnist' или 'cifar')
    
    Returns:
        logging.Logger: Настроенный объект логгера
    """
    logger = logging.getLogger(f'width_experiment_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # Настройка обработчика для записи в файл
    file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, f'{dataset_name}_experiment.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def train_model(model, train_loader, test_loader, config_name, dataset_name, epochs=EPOCHS, lr=LR):
    """
    Функция обучения модели с сохранением метрик.
    
    Args:
        model (nn.Module): Модель для обучения
        train_loader (DataLoader): Загрузчик обучающих данных
        test_loader (DataLoader): Загрузчик тестовых данных
        config_name (str): Название конфигурации
        dataset_name (str): Имя датасета
        epochs (int): Количество эпох
        lr (float): Скорость обучения
    
    Returns:
        dict: История обучения (loss, accuracy, время)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],   # Метрики на обучающей выборке
        'test_loss': [], 'test_acc': [],     # Метрики на тестовой выборке
        'epoch_time': []                     # Время каждой эпохи
    }
    
    logger = logging.getLogger(f'width_experiment_{dataset_name}')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Фаза обучения
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        
        # Фаза тестирования
        model.eval()
        test_loss, test_acc = run_epoch(model, test_loader, criterion, is_test=True)
        
        # Сохранение метрик
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        logger.info(f"Эпоха {epoch+1}/{epochs}: "
                   f"Ошибка (train): {train_loss:.4f}, Ошибка (test): {test_loss:.4f}, "
                   f"Точность (train): {train_acc:.4f}, Точность (test): {test_acc:.4f}, "
                   f"Время: {epoch_time:.2f} сек")
    
    return history

def run_epoch(model, data_loader, criterion, optimizer=None, is_test=False):
    """
    Выполнение одной эпохи обучения или тестирования.
    
    Args:
        model (nn.Module): Модель
        data_loader (DataLoader): Загрузчик данных
        criterion: Функция потерь
        optimizer: Оптимизатор (None для теста)
        is_test (bool): Флаг тестового режима
    
    Returns:
        tuple: (средняя ошибка, точность)
    """
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(data_loader, desc=f"{'Тест' if is_test else 'Обучение'}"):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        if not is_test:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        if not is_test:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# ==================== ОСНОВНЫЕ ФУНКЦИИ ЭКСПЕРИМЕНТОВ ====================

def width_experiment(dataset_name, width_configs, config_names, test_mode=False):
    """
    Эксперимент с различными фиксированными конфигурациями ширины слоев.
    
    Args:
        dataset_name (str): 'mnist' или 'cifar'
        width_configs (list): Список конфигураций ширины (например, [[64,32,16], [256,128,64]])
        config_names (list): Названия конфигураций
        test_mode (bool): Флаг тестового режима (уменьшенные данные)
    
    Returns:
        dict: Результаты всех экспериментов
    """
    logger = setup_logger(dataset_name)
    logger.info(f"Начало эксперимента по ширине для {dataset_name}")
    logger.info(f"Конфигурации: {config_names}")
    
    if test_mode:
        epochs = 2  # В тестовом режиме используем только 2 эпохи
        width_configs = width_configs[:2]
        config_names = config_names[:2]
        logger.info("Активирован тестовый режим")
    else:
        epochs = EPOCHS
    
    # Загрузка данных
    loader_fn = get_mnist_loaders if dataset_name == "mnist" else get_cifar_loaders
    train_loader, test_loader = loader_fn(batch_size=BATCH_SIZE)
    
    # Определение размерности входных данных
    sample, _ = next(iter(train_loader))
    input_size = sample.view(sample.size(0), -1).shape[1]
    num_classes = len(train_loader.dataset.dataset.classes)
    
    results = {}
    
    for i, widths in enumerate(width_configs):
        config_name = config_names[i]
        logger.info(f"\n=== Конфигурация: {config_name} ===")
        logger.info(f"Ширина слоев: {widths}")
        
        # Генерация конфигурации модели
        config = generate_width_config(widths)
        model = create_model(config, input_size, num_classes).to(DEVICE)
        
        param_count = count_parameters(model)
        logger.info(f"Количество параметров модели: {param_count:,}")
        
        # Обучение модели
        history = train_model(
            model, train_loader, test_loader, 
            config_name, dataset_name, epochs
        )
        
        # Сохранение результатов
        results[config_name] = {
            'history': history,  # Метрики обучения
            'widths': widths,    # Использованные ширины слоев
            'params': param_count  # Количество параметров
        }
        
        # Визуализация кривых обучения
        plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_{config_name}.png")
        plot_training_history(
            history, 
            f"{dataset_name} - {config_name}", 
            plot_path
        )
    
    analyze_results(results, dataset_name, logger)
    
    # Сохранение всех результатов в JSON
    results_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Результаты сохранены в {results_path}")
    
    return results

def grid_search_experiment(dataset_name, width_options, depth=3, test_mode=False):
    """
    Эксперимент с поиском по сетке (grid search) различных комбинаций ширины слоев.
    
    Args:
        dataset_name (str): 'mnist' или 'cifar'
        width_options (list): Варианты ширины для поиска (например, [64, 128, 256])
        depth (int): Глубина сети (количество слоев)
        test_mode (bool): Флаг тестового режима
    """
    logger = setup_logger(f"{dataset_name}_grid")
    logger.info(f"Начало grid search для {dataset_name}")
    logger.info(f"Варианты ширины: {width_options}")
    
    if test_mode:
        epochs = 2  # Уменьшенное количество эпох для теста
        width_options = width_options[:2]
        logger.info("Активирован тестовый режим")
    else:
        epochs = 10  # Для grid search используем меньше эпох
    
    # Загрузка данных
    loader_fn = get_mnist_loaders if dataset_name == "mnist" else get_cifar_loaders
    train_loader, test_loader = loader_fn(batch_size=BATCH_SIZE)
    
    # Определение размерности входных данных
    sample, _ = next(iter(train_loader))
    input_size = sample.view(sample.size(0), -1).shape[1]
    num_classes = len(train_loader.dataset.dataset.classes)
    
    # Инициализация матриц для сохранения результатов
    n = len(width_options)
    acc_matrix = np.zeros((n, n))    # Матрица точности
    time_matrix = np.zeros((n, n))   # Матрица времени обучения
    param_matrix = np.zeros((n, n))  # Матрица количества параметров
    
    # Перебор всех комбинаций ширины
    for i, w1 in enumerate(width_options):
        for j, w2 in enumerate(width_options):
            # Формирование архитектуры сети
            if depth == 3:
                widths = [w1, w2, w2 // 2]  # Для 3-слойной сети
            else:
                widths = [w1, w2]           # Для 2-слойной сети
                
            config_name = f"w1_{w1}_w2_{w2}"
            logger.info(f"\n=== Комбинация: {config_name} ===")
            logger.info(f"Ширина слоев: {widths}")
            
            # Создание модели
            config = generate_width_config(widths)
            model = create_model(config, input_size, num_classes).to(DEVICE)
            
            param_count = count_parameters(model)
            logger.info(f"Количество параметров модели: {param_count:,}")
            
            # Обучение модели
            history = train_model(
                model, train_loader, test_loader, 
                config_name, dataset_name, epochs
            )
            
            # Сохранение результатов
            test_acc = history['test_acc'][-1]
            train_time = sum(history['epoch_time'])
            
            acc_matrix[i, j] = test_acc
            time_matrix[i, j] = train_time
            param_matrix[i, j] = param_count
            
            logger.info(f"Точность на тесте: {test_acc:.4f}, Время обучения: {train_time:.1f} сек")
    
    # Сохранение матриц
    np.save(os.path.join(GRID_SEARCH_DIR, f"{dataset_name}_acc_matrix.npy"), acc_matrix)
    np.save(os.path.join(GRID_SEARCH_DIR, f"{dataset_name}_time_matrix.npy"), time_matrix)
    np.save(os.path.join(GRID_SEARCH_DIR, f"{dataset_name}_param_matrix.npy"), param_matrix)
    
    # Визуализация результатов
    row_labels = [str(w) for w in width_options]
    col_labels = [str(w) for w in width_options]
    
    # Heatmap точности
    acc_heatmap_path = os.path.join(PLOTS_DIR, f"{dataset_name}_grid_acc_heatmap.png")
    plot_heatmap(
        acc_matrix, row_labels, col_labels,
        f"{dataset_name}: Точность (heatmap)",
        acc_heatmap_path
    )
    
    # Heatmap времени обучения
    time_heatmap_path = os.path.join(PLOTS_DIR, f"{dataset_name}_grid_time_heatmap.png")
    plot_heatmap(
        time_matrix, row_labels, col_labels,
        f"{dataset_name}: Время обучения (heatmap)",
        time_heatmap_path
    )
    
    # Heatmap количества параметров
    param_heatmap_path = os.path.join(PLOTS_DIR, f"{dataset_name}_grid_param_heatmap.png")
    plot_heatmap(
        param_matrix, row_labels, col_labels,
        f"{dataset_name}: Количество параметров (heatmap)",
        param_heatmap_path
    )
    
    return acc_matrix, time_matrix, param_matrix

def analyze_results(results, dataset_name, logger):
    """
    Анализ и визуализация результатов экспериментов.
    
    Args:
        results (dict): Результаты экспериментов
        dataset_name (str): Имя датасета
        logger: Объект логгера
    """
    config_names = list(results.keys())
    test_accs = [results[name]['history']['test_acc'][-1] for name in config_names]
    train_times = [sum(results[name]['history']['epoch_time']) for name in config_names]
    params = [results[name]['params'] for name in config_names]
    
    # График сравнения точности
    acc_path = os.path.join(PLOTS_DIR, f"{dataset_name}_width_accuracy.png")
    plot_comparison_bar(
        test_accs, config_names, 
        f"{dataset_name}: Точность по конфигурациям",
        "Точность",
        acc_path
    )
    
    # График сравнения времени обучения
    time_path = os.path.join(PLOTS_DIR, f"{dataset_name}_width_time.png")
    plot_comparison_bar(
        train_times, config_names, 
        f"{dataset_name}: Время обучения по конфигурациям",
        "Время (сек)",
        time_path
    )
    
    # График сравнения количества параметров
    param_path = os.path.join(PLOTS_DIR, f"{dataset_name}_width_params.png")
    plot_comparison_bar(
        params, config_names, 
        f"{dataset_name}: Количество параметров по конфигурациям",
        "Параметры",
        param_path
    )
    
    # Вывод сводной информации в лог
    logger.info("\n" + "="*50)
    logger.info(f"{dataset_name.upper()} - Сводные результаты:")
    for name in config_names:
        r = results[name]
        logger.info(f"{name}: Точность (test): {r['history']['test_acc'][-1]:.4f}, "
                   f"Время: {sum(r['history']['epoch_time']):.1f} сек, "
                   f"Параметры: {r['params']:,}")

# ==================== ТОЧКА ВХОДА ====================

def main():
    """Основная функция для запуска экспериментов."""
    parser = argparse.ArgumentParser(description="Эксперименты по исследованию ширины слоев в нейронных сетях")
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True,
                       help="Выбор датасета (MNIST или CIFAR-10)")
    parser.add_argument('--test', action='store_true', 
                       help="Активация тестового режима (уменьшенные данные)")
    parser.add_argument('--grid', action='store_true', 
                       help="Запуск grid search по ширине слоев")
    args = parser.parse_args()
    
    # Фиксированные конфигурации ширины для сравнения
    width_configs = [
        [64, 32, 16],     # Узкая сеть
        [256, 128, 64],    # Средняя сеть
        [1024, 512, 256],  # Широкая сеть
        [2048, 1024, 512]  # Очень широкая сеть
    ]
    config_names = ['narrow', 'medium', 'wide', 'xwide']
    
    # Часть 1: Сравнение фиксированных конфигураций
    width_results = width_experiment(
        args.dataset, width_configs, config_names, test_mode=args.test
    )
    
    # Часть 2: Grid search (только в полном режиме)
    if not args.test and args.grid:
        width_options = [64, 128, 256, 512]  # Варианты ширины для поиска
        grid_search_experiment(
            args.dataset, width_options, depth=3
        )

if __name__ == "__main__":
    main()
