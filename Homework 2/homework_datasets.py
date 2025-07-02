import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2.1 Кастомный Dataset класс
class CustomCSVDataset(Dataset):
    """
    Кастомный датасет для загрузки данных из CSV файла.
    
    Attributes:
        csv_file (str): Путь к CSV файлу.
        target_col (str): Имя целевой колонки.
        features (pd.DataFrame): Признаки.
        target (pd.Series): Целевая переменная.
        numeric_cols (list): Список числовых признаков.
        categorical_cols (list): Список категориальных признаков.
        X_tensor (torch.Tensor): Преобразованные признаки в тензор.
        y_tensor (torch.Tensor): Преобразованная целевая переменная в тензор.
    """
    
    def __init__(self, csv_file, target_col, normalize=True, handle_missing=True, task_type='auto'):
        """
        Args:
            csv_file (str): Путь к CSV файлу.
            target_col (str): Имя целевой колонки.
            normalize (bool): Нужно ли нормализовать числовые признаки.
            handle_missing (bool): Нужно ли обрабатывать пропущенные значения.
            task_type (str): Тип задачи ('classification', 'regression', 'auto')
        """
        self.csv_file = csv_file
        self.target_col = target_col
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.task_type = task_type
        
        # Загрузка данных
        self.df = pd.read_csv(csv_file)
        
        # Обработка пропущенных значений
        if self.handle_missing:
            # Числовые колонки: заполнение средним
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            # Категориальные: заполнение модой
            categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
            mode_values = self.df[categorical_cols].mode().iloc[0]
            self.df[categorical_cols] = self.df[categorical_cols].fillna(mode_values).infer_objects(copy=False)
        
        # Разделение на признаки и целевую переменную
        self.features = self.df.drop(columns=[target_col])
        self.target = self.df[target_col]
        
        # Определение типов признаков
        self.numeric_cols = self.features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.features.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Кодирование категориальных признаков
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.features[col] = le.fit_transform(self.features[col])
            self.label_encoders[col] = le
        
        # Нормализация числовых признаков
        if self.normalize:
            self.scaler = StandardScaler()
            self.features[self.numeric_cols] = self.scaler.fit_transform(self.features[self.numeric_cols])
        
        # Преобразование в тензоры
        self.X_tensor = torch.tensor(self.features.values, dtype=torch.float32)
        
        # Преобразование целевой переменной
        if self.task_type == 'auto':
            if self.target.dtype.name in ['object', 'category']:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'

        if self.task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(self.target)
            self.y_tensor = torch.tensor(y_encoded, dtype=torch.long)
            self.num_classes = len(le.classes_)
        else:
            self.y_tensor = torch.tensor(self.target.values, dtype=torch.float32).squeeze()
            self.num_classes = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

# 2.2 Эксперименты с различными датасетами
if __name__ == "__main__":
    # Тестирование с heart_disease_uci.csv
    logging.info("Загрузка датасета heart_disease_uci.csv")
    heart_dataset = CustomCSVDataset('heart_disease_uci.csv', target_col='num', task_type='classification')
    
    # Преобразование целевой переменной в бинарную
    heart_dataset.y_tensor = (heart_dataset.y_tensor > 0).long()
    heart_dataset.num_classes = 2
    
    # Разделение на train и val
    train_size = int(0.8 * len(heart_dataset))
    val_size = len(heart_dataset) - train_size
    train_dataset, val_dataset = random_split(heart_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Обучение модели логистической регрессии
    input_dim = heart_dataset.X_tensor.shape[1]
    model = MulticlassLogisticRegression(input_dim, num_classes=2)
    logging.info("Обучение модели логистической регрессии")
    train_logistic_regression(model, train_loader, val_loader)
    
    # Сохранение модели логистической регрессии
    logging.info("Сохранение обученной модели логистической регрессии")
    torch.save(model.state_dict(), 'heart_disease_model.pth')
    
    # Тестирование с housing.csv
    logging.info("Загрузка датасета housing.csv")
    housing_dataset = CustomCSVDataset('housing.csv', target_col='median_house_value', task_type='regression')
    
    # Разделение на train и val
    train_size = int(0.8 * len(housing_dataset))
    val_size = len(housing_dataset) - train_size
    train_dataset, val_dataset = random_split(housing_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Обучение модели линейной регрессии
    input_dim = housing_dataset.X_tensor.shape[1]
    model = LinearRegressionWithRegularization(input_dim)
    logging.info("Обучение модели линейной регрессии")
    train_linear_regression(model, train_loader, val_loader)
    
    # Сохранение модели линейной регрессии
    logging.info("Сохранение обученной модели линейной регрессии")
    torch.save(model.state_dict(), 'housing_model.pth')
