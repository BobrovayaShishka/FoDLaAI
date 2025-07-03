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

# Настройка путей
RESULTS_DIR = "results/width_experiments"
GRID_SEARCH_DIR = os.path.join(RESULTS_DIR, "grid_search")
PLOTS_DIR = "plots/width_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRID_SEARCH_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Параметры
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger(dataset_name):
    logger = logging.getLogger(f'width_experiment_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, f'{dataset_name}_experiment.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def train_model(model, train_loader, test_loader, config_name, dataset_name, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': []
    }
    
    logger = logging.getLogger(f'width_experiment_{dataset_name}')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        
        # Testing
        model.eval()
        test_loss, test_acc = run_epoch(model, test_loader, criterion, is_test=True)
        
        # Save metrics
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                   f"Time: {epoch_time:.2f} sec")
    
    return history

def run_epoch(model, data_loader, criterion, optimizer=None, is_test=False):
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(data_loader, desc=f"{'Test' if is_test else 'Train'}"):
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

def width_experiment(dataset_name, width_configs, config_names, test_mode=False):
    logger = setup_logger(dataset_name)
    logger.info(f"Starting width experiment for {dataset_name}")
    logger.info(f"Configurations: {config_names}")
    
    if test_mode:
        epochs = 2
        width_configs = width_configs[:2]
        config_names = config_names[:2]
        logger.info("Test mode activated")
    else:
        epochs = EPOCHS
    
    # Load data
    loader_fn = get_mnist_loaders if dataset_name == "mnist" else get_cifar_loaders
    train_loader, test_loader = loader_fn(batch_size=BATCH_SIZE)
    
    # Get data dimensions
    sample, _ = next(iter(train_loader))
    input_size = sample.view(sample.size(0), -1).shape[1]
    num_classes = len(train_loader.dataset.dataset.classes)
    
    results = {}
    
    for i, widths in enumerate(width_configs):
        config_name = config_names[i]
        logger.info(f"\n=== Configuration: {config_name} ===")
        logger.info(f"Layer widths: {widths}")
        
        # Generate config
        config = generate_width_config(widths)
        model = create_model(config, input_size, num_classes).to(DEVICE)
        
        param_count = count_parameters(model)
        logger.info(f"Model parameters: {param_count:,}")
        
        # Train model
        history = train_model(
            model, train_loader, test_loader, 
            config_name, dataset_name, epochs
        )
        
        # Save results
        results[config_name] = {
            'history': history,
            'widths': widths,
            'params': param_count
        }
        
        # Visualization
        plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_{config_name}.png")
        plot_training_history(
            history, 
            f"{dataset_name} - {config_name}", 
            plot_path
        )
    
    # Analyze results
    analyze_results(results, dataset_name, logger)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results

def analyze_results(results, dataset_name, logger):
    config_names = list(results.keys())
    test_accs = [results[name]['history']['test_acc'][-1] for name in config_names]
    train_times = [sum(results[name]['history']['epoch_time']) for name in config_names]
    params = [results[name]['params'] for name in config_names]
    
    # Accuracy comparison
    acc_path = os.path.join(PLOTS_DIR, f"{dataset_name}_width_accuracy.png")
    plot_comparison_bar(
        test_accs, config_names, 
        f"{dataset_name}: Accuracy by Width Configuration",
        "Accuracy",
        acc_path
    )
    
    # Time comparison
    time_path = os.path.join(PLOTS_DIR, f"{dataset_name}_width_time.png")
    plot_comparison_bar(
        train_times, config_names, 
        f"{dataset_name}: Training Time by Width Configuration",
        "Time (sec)",
        time_path
    )
    
    # Parameters comparison
    param_path = os.path.join(PLOTS_DIR, f"{dataset_name}_width_params.png")
    plot_comparison_bar(
        params, config_names, 
        f"{dataset_name}: Parameters by Width Configuration",
        "Parameters",
        param_path
    )
    
    # Log summary
    logger.info("\n" + "="*50)
    logger.info(f"{dataset_name.upper()} - Results Summary:")
    for name in config_names:
        r = results[name]
        logger.info(f"{name}: Test Acc: {r['history']['test_acc'][-1]:.4f}, "
                   f"Time: {sum(r['history']['epoch_time']):.1f} sec, "
                   f"Params: {r['params']:,}")

def grid_search_experiment(dataset_name, width_options, depth=3, test_mode=False):
    logger = setup_logger(f"{dataset_name}_grid")
    logger.info(f"Starting grid search for {dataset_name}")
    logger.info(f"Width options: {width_options}")
    
    if test_mode:
        epochs = 2
        width_options = width_options[:2]
        logger.info("Test mode activated")
    else:
        epochs = 10
    
    # Load data
    loader_fn = get_mnist_loaders if dataset_name == "mnist" else get_cifar_loaders
    train_loader, test_loader = loader_fn(batch_size=BATCH_SIZE)
    
    # Get data dimensions
    sample, _ = next(iter(train_loader))
    input_size = sample.view(sample.size(0), -1).shape[1]
    num_classes = len(train_loader.dataset.dataset.classes)
    
    # Initialize matrices
    n = len(width_options)
    acc_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))
    param_matrix = np.zeros((n, n))
    
    for i, w1 in enumerate(width_options):
        for j, w2 in enumerate(width_options):
            if depth == 3:
                widths = [w1, w2, w2 // 2]
            else:
                widths = [w1, w2]
                
            config_name = f"w1_{w1}_w2_{w2}"
            logger.info(f"\n=== Grid: {config_name} ===")
            logger.info(f"Layer widths: {widths}")
            
            # Generate config
            config = generate_width_config(widths)
            model = create_model(config, input_size, num_classes).to(DEVICE)
            
            param_count = count_parameters(model)
            logger.info(f"Model parameters: {param_count:,}")
            
            # Train model
            history = train_model(
                model, train_loader, test_loader, 
                config_name, dataset_name, epochs
            )
            
            # Save results
            test_acc = history['test_acc'][-1]
            train_time = sum(history['epoch_time'])
            
            acc_matrix[i, j] = test_acc
            time_matrix[i, j] = train_time
            param_matrix[i, j] = param_count
            
            logger.info(f"Test Acc: {test_acc:.4f}, Time: {train_time:.1f} sec")
    
    # Save matrices
    np.save(os.path.join(GRID_SEARCH_DIR, f"{dataset_name}_acc_matrix.npy"), acc_matrix)
    np.save(os.path.join(GRID_SEARCH_DIR, f"{dataset_name}_time_matrix.npy"), time_matrix)
    np.save(os.path.join(GRID_SEARCH_DIR, f"{dataset_name}_param_matrix.npy"), param_matrix)
    
    # Visualize results
    row_labels = [str(w) for w in width_options]
    col_labels = [str(w) for w in width_options]
    
    acc_heatmap_path = os.path.join(PLOTS_DIR, f"{dataset_name}_grid_acc_heatmap.png")
    plot_heatmap(
        acc_matrix, row_labels, col_labels,
        f"{dataset_name}: Accuracy Heatmap",
        acc_heatmap_path
    )
    
    time_heatmap_path = os.path.join(PLOTS_DIR, f"{dataset_name}_grid_time_heatmap.png")
    plot_heatmap(
        time_matrix, row_labels, col_labels,
        f"{dataset_name}: Training Time Heatmap (sec)",
        time_heatmap_path
    )
    
    param_heatmap_path = os.path.join(PLOTS_DIR, f"{dataset_name}_grid_param_heatmap.png")
    plot_heatmap(
        param_matrix, row_labels, col_labels,
        f"{dataset_name}: Parameters Heatmap",
        param_heatmap_path
    )
    
    return acc_matrix, time_matrix, param_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True)
    parser.add_argument('--test', action='store_true', help="Test mode")
    args = parser.parse_args()
    
    # Fixed width configurations
    width_configs = [
        [64, 32, 16],    # narrow
        [256, 128, 64],  # medium
        [1024, 512, 256],# wide
        [2048, 1024, 512]# x-wide
    ]
    config_names = ['narrow', 'medium', 'wide', 'xwide']
    
    # Part 1: Fixed width experiments
    width_results = width_experiment(
        args.dataset, width_configs, config_names, test_mode=args.test
    )
    
    # Part 2: Grid search
    if not args.test:
        width_options = [64, 128, 256, 512]
        grid_search_experiment(
            args.dataset, width_options, depth=3
        )

if __name__ == "__main__":
    main()
