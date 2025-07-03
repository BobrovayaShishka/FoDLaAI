import json
import torch
import torch.nn as nn

def create_model(config, input_size, num_classes):
    """Создание модели из конфигурации"""
    layers = []
    prev_size = input_size
    
    for layer_spec in config['layers']:
        layer_type = layer_spec['type']
        
        if layer_type == 'linear':
            out_size = layer_spec['size']
            layers.append(nn.Linear(prev_size, out_size))
            prev_size = out_size
            
        elif layer_type == 'relu':
            layers.append(nn.ReLU())
            
        elif layer_type == 'sigmoid':
            layers.append(nn.Sigmoid())
            
        elif layer_type == 'tanh':
            layers.append(nn.Tanh())
            
        elif layer_type == 'dropout':
            rate = layer_spec.get('rate', 0.5)
            layers.append(nn.Dropout(rate))
            
        elif layer_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(prev_size))
            
        elif layer_type == 'layer_norm':
            layers.append(nn.LayerNorm(prev_size))
    
    layers.append(nn.Linear(prev_size, num_classes))
    
    return nn.Sequential(*layers)

def count_parameters(model):
    """Подсчет обучаемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_depth_config(depth, hidden_size=256, use_dropout=False, use_batchnorm=False):
    """Генерация конфигурации по глубине"""
    layers = []
    for i in range(depth - 1):
        layers.append({"type": "linear", "size": hidden_size})
        
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        
        layers.append({"type": "relu"})
        
        if use_dropout:
            layers.append({"type": "dropout", "rate": 0.5})
    
    return {"layers": layers}

def generate_width_config(widths, use_dropout=False, use_batchnorm=False):
    """Генерация конфигурации по ширине"""
    layers = []
    for i, width in enumerate(widths):
        layers.append({"type": "linear", "size": width})
        
        if use_batchnorm and i < len(widths) - 1:  # No BN before output
            layers.append({"type": "batch_norm"})
        
        if i < len(widths) - 1:  # No activation before output
            layers.append({"type": "relu"})
        
        if use_dropout and i > 0:  # No dropout after input
            layers.append({"type": "dropout", "rate": 0.5})
    
    return {"layers": layers}

def generate_config(widths, dropout_rate=0.0, use_batchnorm=False, 
                   weight_decay=0.0, adaptive_dropout=False, 
                   bn_momentum=0.1, layer_specific=False):
    """Генерация конфигурации с регуляризацией"""
    layers = []
    
    for i, width in enumerate(widths):
        layers.append({"type": "linear", "size": width})
        
        # BatchNorm
        if use_batchnorm:
            if not layer_specific or (layer_specific and i < len(widths)-1):
                layers.append({"type": "batch_norm"})
        
        # Activation (except last layer)
        if i < len(widths) - 1:
            layers.append({"type": "relu"})
        
        # Dropout
        if dropout_rate > 0:
            if not layer_specific or (layer_specific and i > 0):
                layers.append({"type": "dropout", "rate": dropout_rate})
    
    return {
        "layers": layers,
        "dropout_rate": dropout_rate,
        "use_batchnorm": use_batchnorm,
        "weight_decay": weight_decay,
        "adaptive_dropout": adaptive_dropout,
        "bn_momentum": bn_momentum,
        "layer_specific": layer_specific
    }
