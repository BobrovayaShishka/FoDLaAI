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
from utils.model_utils import generate_depth_config, create_model, count_parameters
from utils.visualization_utils import plot_training_history, plot_comparison_bar

# Настройка путей
RESULTS_DIR = "results/depth_experiments"
MODEL_CONFIGS_DIR = os.path.join(RESULTS_DIR, "model_configs")
WEIGHT_STATS_DIR = os.path.join(RESULTS_DIR, "weight_stats")
PLOTS_DIR = "plots/depth_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_CONFIGS_DIR, exist_ok=True)
os.makedirs(WEIGHT_STATS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Параметры
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger(dataset_name):
    """Настройка логгера"""
    logger = logging.getLogger(f'depth_experiment_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # Создаем обработчик для записи в файл
    file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, f'{dataset_name}_experiment.log'))
    file_handler.setLevel(logging.INFO)
    
    # Создаем форматтер
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)
    
    return logger

def save_model_config(config, depth, dataset_name):
    """Сохраняет конфигурацию модели в JSON"""
    config_path = os.path.join(MODEL_CONFIGS_DIR, f"{dataset_name}_depth_{depth}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

def save_weight_stats(model, epoch, depth, dataset_name):
    """Сохраняет статистику весов модели"""
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            }
    
    stats_path = os.path.join(WEIGHT_STATS_DIR, f"{dataset_name}_depth_{depth}_epoch_{epoch}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats_path

def run_epoch(model, data_loader, criterion, optimizer=None, is_test=False):
    """Запуск одной эпохи обучения/тестирования"""
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(data_loader, desc=f"{'Test' if is_test else 'Train'} Epoch"):
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

def train_model(model, train_loader, test_loader, depth, dataset_name, epochs=EPOCHS, lr=LR):
    """Полный цикл обучения модели"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': []
    }
    
    logger = logging.getLogger(f'depth_experiment_{dataset_name}')
    
    for epoch in range(epochs):
        start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Обучение
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        
        # Тестирование
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, is_test=True
        )
        
        # Сохранение статистики весов
        save_weight_stats(model, epoch+1, depth, dataset_name)
        
        # Сохранение метрик
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_time:.2f} sec")
    
    return history

def run_depth_experiment(dataset_name, depths, use_dropout=False, use_batchnorm=False, test_mode=False):
    """Эксперимент с различной глубиной сети"""
    # Настройка логгера
    logger = setup_logger(dataset_name)
    logger.info(f"Starting depth experiment for {dataset_name}")
    logger.info(f"Depths: {depths}, Dropout: {use_dropout}, BatchNorm: {use_batchnorm}")
    
    if test_mode:
        epochs = 2
        depths = depths[:2]
        logger.info("Test mode activated: reduced epochs and depths")
    else:
        epochs = EPOCHS
    
    # Загрузка данных
    loader_fn = get_mnist_loaders if dataset_name == "mnist" else get_cifar_loaders
    train_loader, test_loader = loader_fn(batch_size=BATCH_SIZE)
    
    # Определение размерности данных
    sample, _ = next(iter(train_loader))
    input_size = sample.view(sample.size(0), -1).shape[1]
    num_classes = len(train_loader.dataset.dataset.classes)
    
    results = {}
    
    for depth in depths:
        logger.info(f"\n=== Depth: {depth} layers ===")
        
        # Генерация конфигурации
        config = generate_depth_config(depth, HIDDEN_SIZE, use_dropout, use_batchnorm)
        save_model_config(config, depth, dataset_name)
        
        model = create_model(config, input_size, num_classes).to(DEVICE)
        
        param_count = count_parameters(model)
        logger.info(f"Model parameters: {param_count:,}")
        
        # Обучение модели
        history = train_model(
            model, train_loader, test_loader, 
            depth, dataset_name, epochs
        )
        
        # Сохранение результатов
        results[depth] = {
            'history': history,
            'params': param_count,
            'config': config
        }
        
        # Визуализация
        plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_depth_{depth}.png")
        plot_training_history(
            history, 
            f"{dataset_name} - Depth: {depth}", 
            plot_path
        )
    
    # Анализ результатов
    analyze_results(results, dataset_name, logger)
    
    # Сохранение результатов
    results_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results

def analyze_results(results, dataset_name, logger):
    """Анализ и сравнение результатов"""
    depths = list(results.keys())
    test_accs = [results[d]['history']['test_acc'][-1] for d in depths]
    train_times = [sum(results[d]['history']['epoch_time']) for d in depths]
    
    # Визуализация сравнения
    acc_path = os.path.join(PLOTS_DIR, f"{dataset_name}_depth_accuracy.png")
    plot_comparison_bar(
        test_accs, [str(d) for d in depths], 
        f"{dataset_name}: Accuracy by Depth",
        "Accuracy",
        acc_path
    )
    
    time_path = os.path.join(PLOTS_DIR, f"{dataset_name}_depth_time.png")
    plot_comparison_bar(
        train_times, [str(d) for d in depths], 
        f"{dataset_name}: Training Time by Depth",
        "Time (sec)",
        time_path
    )
    
    # Логирование результатов
    logger.info("\n" + "="*50)
    logger.info(f"{dataset_name.upper()} - Results Summary:")
    for depth in depths:
        h = results[depth]['history']
        logger.info(f"Depth {depth}: Test Acc: {h['test_acc'][-1]:.4f}, Time: {sum(h['epoch_time']):.1f} sec, Params: {results[depth]['params']:,}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True)
    parser.add_argument('--test', action='store_true', help="Test mode")
    args = parser.parse_args()
    
    depths = [1, 2, 3, 5, 7]
    
    # Базовые эксперименты
    base_results = run_depth_experiment(
        args.dataset, depths, test_mode=args.test
    )
    
    # Эксперименты с регуляризацией
    if not args.test:
        reg_results = run_depth_experiment(
            args.dataset, [5, 7], 
            use_dropout=True, use_batchnorm=True
        )

if __name__ == "__main__":
    main()
