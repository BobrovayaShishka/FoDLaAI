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

# Настройка путей
RESULTS_DIR = "results/regularization_experiments"
ADAPTIVE_DIR = os.path.join(RESULTS_DIR, "adaptive")
WEIGHT_DIST_DIR = os.path.join(RESULTS_DIR, "weight_distributions")
PLOTS_DIR = "plots/regularization_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ADAPTIVE_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIST_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Параметры
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_WIDTHS = [1024, 512, 256]

def setup_logger(dataset_name):
    logger = logging.getLogger(f'reg_experiment_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, f'{dataset_name}_experiment.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def run_epoch(model, data_loader, criterion, optimizer=None, is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
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

def train_model(model, train_loader, test_loader, config, exp_name, dataset_name, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': [],
        'weight_stats': []
    }
    
    logger = logging.getLogger(f'reg_experiment_{dataset_name}')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Adaptive dropout
        if config.get('adaptive_dropout', False):
            current_dropout = config['dropout_rate'] * (epoch / epochs)
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = current_dropout
            logger.info(f"Epoch {epoch+1}: Adaptive dropout rate = {current_dropout:.3f}")
        
        # Training
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        
        # Testing
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, is_test=True
        )
        
        # Save metrics
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        # Weight statistics
        weight_stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' not in name:
                weight_stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'abs_mean': param.data.abs().mean().item()
                }
        history['weight_stats'].append(weight_stats)
        
        # Plot weight distribution every 5 epochs
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
    
    # Save weight statistics
    weight_stats_path = os.path.join(WEIGHT_DIST_DIR, f"{exp_name}_weight_stats.json")
    with open(weight_stats_path, 'w') as f:
        json.dump(history['weight_stats'], f, indent=2)
    
    return history

def regularization_experiment(dataset_name, test_mode=False):
    logger = setup_logger(dataset_name)
    logger.info(f"Starting regularization experiment for {dataset_name}")
    
    if test_mode:
        epochs = 2
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
    
    # Part 1: Regularization techniques comparison
    experiments = [
        {"name": "no_reg", "config": generate_config(BASE_WIDTHS)},
        {"name": "dropout_0.1", "config": generate_config(BASE_WIDTHS, dropout_rate=0.1)},
        {"name": "dropout_0.3", "config": generate_config(BASE_WIDTHS, dropout_rate=0.3)},
        {"name": "dropout_0.5", "config": generate_config(BASE_WIDTHS, dropout_rate=0.5)},
        {"name": "batchnorm", "config": generate_config(BASE_WIDTHS, use_batchnorm=True)},
        {"name": "both", "config": generate_config(BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True)},
        {"name": "l2_1e-4", "config": generate_config(BASE_WIDTHS, weight_decay=1e-4)},
        {"name": "l2_1e-3", "config": generate_config(BASE_WIDTHS, weight_decay=1e-3)},
        {"name": "l2_1e-2", "config": generate_config(BASE_WIDTHS, weight_decay=1e-2)},
    ]
    
    for exp in experiments:
        logger.info(f"\n=== Experiment: {exp['name']} ===")
        
        # Create model
        model = create_model(exp['config'], input_size, num_classes).to(DEVICE)
        
        # Set batch norm momentum if needed
        if exp['config'].get('use_batchnorm', False):
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.momentum = exp['config'].get('bn_momentum', 0.1)
        
        param_count = count_parameters(model)
        logger.info(f"Model parameters: {param_count:,}")
        
        # Train model
        history = train_model(
            model, train_loader, test_loader, 
            exp['config'], exp['name'], dataset_name, epochs
        )
        
        # Save results
        results[exp['name']] = {
            'history': history,
            'config': exp['config'],
            'params': param_count
        }
        
        # Visualization
        plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_{exp['name']}.png")
        plot_training_history(
            history, 
            f"{dataset_name} - {exp['name']}", 
            plot_path
        )
    
    # Part 2: Adaptive regularization techniques
    if not test_mode:
        adaptive_experiments = [
            {"name": "adaptive_dropout", "config": generate_config(
                BASE_WIDTHS, dropout_rate=0.5, adaptive_dropout=True
            )},
            {"name": "bn_momentum_0.5", "config": generate_config(
                BASE_WIDTHS, use_batchnorm=True, bn_momentum=0.5
            )},
            {"name": "bn_momentum_0.9", "config": generate_config(
                BASE_WIDTHS, use_batchnorm=True, bn_momentum=0.9
            )},
            {"name": "combined", "config": generate_config(
                BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True, weight_decay=1e-4
            )},
            {"name": "layer_specific", "config": generate_config(
                BASE_WIDTHS, dropout_rate=0.3, use_batchnorm=True, layer_specific=True
            )},
        ]
        
        for exp in adaptive_experiments:
            logger.info(f"\n=== Adaptive Experiment: {exp['name']} ===")
            
            # Create model
            model = create_model(exp['config'], input_size, num_classes).to(DEVICE)
            
            # Set batch norm momentum
            if exp['config'].get('use_batchnorm', False):
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.momentum = exp['config'].get('bn_momentum', 0.1)
            
            param_count = count_parameters(model)
            logger.info(f"Model parameters: {param_count:,}")
            
            # Train model
            history = train_model(
                model, train_loader, test_loader, 
                exp['config'], exp['name'], dataset_name, epochs
            )
            
            # Save results
            results[exp['name']] = {
                'history': history,
                'config': exp['config'],
                'params': param_count
            }
            
            # Save adaptive results separately
            adaptive_path = os.path.join(ADAPTIVE_DIR, f"{dataset_name}_{exp['name']}.json")
            with open(adaptive_path, 'w') as f:
                json.dump({
                    'history': history,
                    'config': exp['config'],
                    'params': param_count
                }, f, indent=2)
            
            # Visualization
            plot_path = os.path.join(PLOTS_DIR, f"{dataset_name}_{exp['name']}.png")
            plot_training_history(
                history, 
                f"{dataset_name} - {exp['name']}", 
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
    names = list(results.keys())
    test_accs = [results[name]['history']['test_acc'][-1] for name in names]
    train_times = [sum(results[name]['history']['epoch_time']) for name in names]
    
    # Accuracy comparison
    acc_path = os.path.join(PLOTS_DIR, f"{dataset_name}_comparison.png")
    plot_comparison_bar(
        test_accs, names, 
        f"{dataset_name}: Regularization Techniques Comparison",
        "Accuracy",
        acc_path
    )
    
    # Log summary
    logger.info("\n" + "="*50)
    logger.info(f"{dataset_name.upper()} - Results Summary:")
    for name in names:
        r = results[name]
        logger.info(f"{name}: Test Acc: {r['history']['test_acc'][-1]:.4f}, "
                   f"Time: {sum(r['history']['epoch_time']):.1f} sec, "
                   f"Params: {r['params']:,}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], required=True)
    parser.add_argument('--test', action='store_true', help="Test mode")
    args = parser.parse_args()
    
    regularization_experiment(args.dataset, test_mode=args.test)

if __name__ == "__main__":
    main()
