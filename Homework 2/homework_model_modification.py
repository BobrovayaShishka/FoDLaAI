import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1.1 Расширение линейной регрессии
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

def train_linear_regression(
    model, 
    X_train, y_train, 
    X_val, y_val,
    epochs=100,
    batch_size=32,
    lr=0.01,
    patience=5
):
    """Обучение модели с ранней остановкой"""
    # Создание DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Валидация
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        # Логирование
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")
        
        # Ранняя остановка
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # График потерь
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('linear_regression_loss.png')
    plt.close()
    
    return model
    
# 1.2 Расширение логистической регрессии
class MulticlassLogisticRegression(nn.Module):
    """Логистическая регрессия для многоклассовой классификации"""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def train_logistic_regression(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    lr=0.01
):
    """Обучение модели с вычислением метрик"""
    # Создание DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    all_labels = []
    all_probs = []
    train_losses, val_losses = [], []
    metrics = {
        'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.squeeze().long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Валидация и метрики
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val.squeeze().long()).item()
            
            # Прогнозы для метрик
            probs = torch.softmax(val_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(y_val.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Вычисление метрик
            precision = precision_score(y_val, preds, average='macro', zero_division=0)
            recall = recall_score(y_val, preds, average='macro')
            f1 = f1_score(y_val, preds, average='macro')
            auc = roc_auc_score(
                y_val, 
                probs.cpu().numpy(), 
                multi_class='ovr', 
                average='macro'
            )
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['auc'].append(auc)
        
        # Логирование
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f}"
        )
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(model.linear.out_features))
    plt.yticks(np.arange(model.linear.out_features))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Графики метрик
    plt.figure(figsize=(12, 8))
    for i, (metric, values) in enumerate(metrics.items()):
        plt.subplot(2, 2, i+1)
        plt.plot(values)
        plt.title(metric.capitalize())
        plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('classification_metrics.png')
    plt.close()
    
    return model

# Тестирование
if __name__ == "__main__":
    # Генерация данных для регрессии
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Линейная регрессия
    lin_model = LinearRegressionWithRegularization(5, l1_lambda=0.01, l2_lambda=0.01)
    train_linear_regression(lin_model, X_train, y_train, X_val, y_val)
    
    # Генерация данных для классификации
    X, y = make_classification(
        n_samples=1000, n_features=10, n_classes=3, n_informative=4
    )
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Логистическая регрессия
    log_model = MulticlassLogisticRegression(10, num_classes=3)
    train_logistic_regression(log_model, X_train, y_train, X_val, y_val)
