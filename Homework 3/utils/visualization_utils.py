import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

def plot_training_history(history, title, save_path):
    """Визуализация истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['test_loss'], label='Test')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['test_acc'], label='Test')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_weight_distribution(model, epoch, title, save_path):
    """Визуализация распределения весов"""
    plt.figure(figsize=(10, 6))
    
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            weights = param.data.cpu().numpy().flatten()
            plt.hist(weights, bins=50, alpha=0.5, label=name)
    
    plt.title(f"{title}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_comparison_bar(data, labels, title, ylabel, save_path):
    """Столбчатая диаграмма для сравнения"""
    plt.figure(figsize=(12, 6))
    plt.bar(labels, data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(matrix, row_labels, col_labels, title, save_path):
    """Визуализация heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".4f", 
                xticklabels=col_labels, 
                yticklabels=row_labels,
                cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
