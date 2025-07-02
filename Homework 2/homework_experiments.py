# homework_experiments.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
import logging
import pandas as pd
import seaborn as sns

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Базовая модель из предыдущего задания
class LinearRegressionWithRegularization(nn.Module):
    """Линейная регрессия с поддержкой L1 и L2 регуляризации"""
    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


def train_linear_regression(model, X_train, y_train, X_val, y_val,
                            epochs=100, batch_size=32, lr=0.01,
                            patience=5, optimizer_type='SGD'):
    """Обучение модели с ранней остановкой и разными оптимизаторами"""

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_type}")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Ранняя остановка на эпохе № {epoch+1}")
                break

    return train_losses, val_losses


# 3.1 Исследование гиперпараметров
def hyperparameter_experiment():
    """Эксперименты с различными гиперпараметрами"""
    # Генерация данных
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']
    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            for opt in optimizers:
                logging.info(f"Эксперимент: LR={lr}, Batch={bs}, Optimizer={opt}")
                model = LinearRegressionWithRegularization(5)
                train_losses, val_losses = train_linear_regression(
                    model, X_train, y_train, X_val, y_val,
                    epochs=100, batch_size=bs, lr=lr, optimizer_type=opt
                )
                results.append({
                    'learning_rate': lr,
                    'batch_size': bs,
                    'optimizer': opt,
                    'best_val_loss': min(val_losses),
                    'epochs_run': len(train_losses)
                })

                # График потерь
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.title(f'LR={lr}, Batch={bs}, Optimizer={opt}')
                plt.xlabel('Epochs')
                plt.ylabel('MSE Loss')
                plt.legend()
                plt.savefig(f'loss_{lr}_{bs}_{opt}.png')
                plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_results.csv', index=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='best_val_loss', y='optimizer', data=results_df)
    plt.title('Сравнение оптимизаторов')
    plt.savefig('optimizer_comparison.png')
    plt.close()

    return results_df


# 3.2 Feature Engineering
def feature_engineering_experiment():
    """Эксперименты с добавлением новых признаков"""
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Статистические признаки
    X_train_stats = np.column_stack((X_train,
                                    np.mean(X_train, axis=1),
                                    np.var(X_train, axis=1)))
    X_val_stats = np.column_stack((X_val,
                                   np.mean(X_val, axis=1),
                                   np.var(X_val, axis=1)))

    # Преобразование в тензоры
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_train_poly_tensor = torch.tensor(X_train_poly, dtype=torch.float32)
    X_val_poly_tensor = torch.tensor(X_val_poly, dtype=torch.float32)
    X_train_stats_tensor = torch.tensor(X_train_stats, dtype=torch.float32)
    X_val_stats_tensor = torch.tensor(X_val_stats, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    results = []

    for name, X_train_feat, X_val_feat in [
        ('Base', X_train_tensor, X_val_tensor),
        ('Polynomial', X_train_poly_tensor, X_val_poly_tensor),
        ('With Stats', X_train_stats_tensor, X_val_stats_tensor)
    ]:
        logging.info(f"Обучение модели с признаками: {name}")
        model = LinearRegressionWithRegularization(X_train_feat.shape[1])
        train_losses, val_losses = train_linear_regression(
            model, X_train_feat, y_train_tensor, X_val_feat, y_val_tensor,
            epochs=100, batch_size=32, lr=0.01
        )

        results.append({
            'feature_type': name,
            'best_val_loss': min(val_losses),
            'final_train_loss': train_losses[-1]
        })

        # Графики потерь
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Модель: {name}')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(f'loss_{name}.png')
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv('feature_engineering_results.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='feature_type', y='best_val_loss', data=results_df)
    plt.title('Сравнение методов feature engineering')
    plt.ylabel('Лучшая validation loss')
    plt.savefig('feature_comparison.png')
    plt.close()

    return results_df


# Запуск экспериментов
if __name__ == "__main__":
    logging.info("Начало экспериментов с гиперпараметрами")
    hyperparameter_results = hyperparameter_experiment()
    print("Результаты экспериментов с гиперпараметрами:")
    print(hyperparameter_results)

    logging.info("Начало экспериментов с feature engineering")
    feature_results = feature_engineering_experiment()
    print("Результаты экспериментов с feature engineering:")
    print(feature_results)
