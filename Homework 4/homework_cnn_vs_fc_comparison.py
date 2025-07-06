import time
import logging
import torch
import numpy as np
from models import fc_models, cnn_models
from datasets import get_mnist_loaders, get_cifar_loaders
from utils.training_utils import train_model, count_parameters, run_epoch
from utils.visualization_utils import (
    plot_training_history, 
    plot_confusion_matrix,
    plot_model_comparison
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def compare_mnist_models():
    """Сравнение моделей на MNIST"""
    logger.info("\n" + "="*50)
    logger.info("Starting MNIST comparison")
    logger.info("="*50)
    save_dir = "results/mnist_comparison"
    
    # Гиперпараметры
    batch_size = 64
    epochs = 10
    lr = 0.001

    # Загрузка данных
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Модели для тестирования
    models = {
        "FC": fc_models.FCNetworkMNIST().to(device),
        "SimpleCNN": cnn_models.SimpleCNN().to(device),
        "ResCNN": cnn_models.CNNWithResidual().to(device)
    }

    results = {}
    for name, model in models.items():
        logger.info(f"\nTraining {name} model...")
        logger.info(f"Number of parameters: {count_parameters(model):,}")

        # Тренировка с замером времени
        start_time = time.time()
        history = train_model(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        train_time = time.time() - start_time

        # Тестирование с замером времени
        start_inference = time.time()
        test_loss, test_acc = run_epoch(
            model, test_loader,
            torch.nn.CrossEntropyLoss(),
            device=device, is_test=True
        )
        inference_time = time.time() - start_inference

        # Сохранение результатов
        results[name] = {
            "history": history,
            "train_time": train_time,
            "inference_time": inference_time,
            "test_acc": test_acc,
            "params": count_parameters(model)
        }

        # Визуализация кривых обучения
        plot_training_history(history, f"{name} - MNIST Training", save_dir)

    # Сравнение результатов
    logger.info("\nMNIST Results Comparison:")
    for name, res in results.items():
        logger.info(
            f"{name}: "
            f"Test Acc = {res['test_acc']:.4f}, "
            f"Train Time = {res['train_time']:.2f}s, "
            f"Inference Time = {res['inference_time']:.2f}s, "
            f"Params = {res['params']:,}"
        )

    # Визуализация сравнения точности
    histories = [res["history"] for res in results.values()]
    labels = list(results.keys())
    plot_model_comparison(
        histories, labels, 
        metric='test_accs',
        title="MNIST Test Accuracy Comparison",
        save_dir=save_dir
    )

def compare_cifar_models():
    """Сравнение моделей на CIFAR-10"""
    logger.info("\n" + "="*50)
    logger.info("Starting CIFAR-10 comparison")
    logger.info("="*50)
    save_dir = "results/cifar_comparison"
    
    # Гиперпараметры
    batch_size = 128
    epochs = 10
    lr = 0.001

    # Загрузка данных
    train_loader, test_loader = get_cifar_loaders(batch_size)

    # Модели для тестирования
    models = {
        "FC": fc_models.FCNetworkCIFAR().to(device),
        "CNN": cnn_models.CIFARCNN().to(device),
        "ResCNN": cnn_models.ResidualCIFAR().to(device)
    }

    results = {}
    for name, model in models.items():
        logger.info(f"\nTraining {name} model...")
        logger.info(f"Number of parameters: {count_parameters(model):,}")

        # Тренировка с замером времени
        start_time = time.time()
        history = train_model(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        train_time = time.time() - start_time

        # Тестирование
        start_inference = time.time()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion,
            device=device, is_test=True
        )
        inference_time = time.time() - start_inference

        # Confusion matrix
        all_preds = []
        all_targets = []
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
        plot_confusion_matrix(
            all_targets, all_preds, classes,
            f"{name} - CIFAR-10 Confusion Matrix",
            save_dir
        )

        # Сохранение результатов
        results[name] = {
            "history": history,
            "train_time": train_time,
            "inference_time": inference_time,
            "test_acc": test_acc,
            "params": count_parameters(model),
        }

        # Визуализация кривых обучения
        plot_training_history(history, f"{name} - CIFAR-10 Training", save_dir)

    # Сравнение результатов
    logger.info("\nCIFAR-10 Results Comparison:")
    for name, res in results.items():
        logger.info(
            f"{name}: "
            f"Test Acc = {res['test_acc']:.4f}, "
            f"Train Time = {res['train_time']:.2f}s, "
            f"Inference Time = {res['inference_time']:.2f}s, "
            f"Params = {res['params']:,}"
        )

    # Визуализация сравнения точности
    histories = [res["history"] for res in results.values()]
    labels = list(results.keys())
    plot_model_comparison(
        histories, labels, 
        metric='test_accs',
        title="CIFAR-10 Test Accuracy Comparison",
        save_dir=save_dir
    )

    # Анализ переобучения (сравнение train/test loss)
    plot_model_comparison(
        histories, labels, 
        metric='train_losses',
        title="CIFAR-10 Train Loss Comparison",
        save_dir=save_dir
    )
    plot_model_comparison(
        histories, labels, 
        metric='test_losses',
        title="CIFAR-10 Test Loss Comparison",
        save_dir=save_dir
    )

if __name__ == "__main__":
    compare_mnist_models()
    compare_cifar_models()
