import os
import json
import time
import logging
import torch
from tqdm import tqdm

def setup_logger(name, log_file, level=logging.INFO):
    """Настройка логгера"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def save_experiment_results(results, output_dir, filename):
    """Сохранение результатов эксперимента"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Результаты сохранены в {path}")

def run_epoch(model, data_loader, criterion, optimizer, device, logger, is_test=False):
    """Запуск одной эпохи обучения/тестирования"""
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        
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
    
    logger.info(f"{'Test' if is_test else 'Train'} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, epochs, lr, device, logger):
    """Полный цикл обучения модели"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Обучение
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # Тестирование
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, None, device, logger, is_test=True
        )
        
        # Сохранение метрик
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        logger.info(f"Epoch {epoch+1} завершена за {epoch_time:.2f} сек")
    
    return history
