import torch
import time
import numpy as np
from collections import defaultdict

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}\n")

def create_tensors():
    """
    Создает три больших тензора заданных размеров.
    Возвращает словарь с тензорами на CPU и GPU (если доступно).
    """
    tensor_shapes = [
        (64, 1024, 1024),    # 64 матрицы 1024x1024
        (128, 512, 512),     # 128 матриц 512x512
        (256, 256, 256)      # 256 матриц 256x256
    ]
    
    tensors = {"cpu": {}, "gpu": {}}
    
    for shape in tensor_shapes:
        # Генерация случайных данных
        data = torch.randn(*shape)
        
        # Сохранение на CPU
        tensors["cpu"][shape] = data.clone()
        
        # Копирование на GPU при наличии
        if device.type == "cuda":
            tensors["gpu"][shape] = data.to(device)
    
    return tensors

def measure_time(operation, tensors, device_type, num_repeats=10):
    """
    Измеряет среднее время выполнения операции.
    
    Параметры:
        operation: функция (a, b) -> result
        tensors: словарь с тензорами
        device_type: "cpu" или "gpu"
        num_repeats: количество повторений
    
    Возвращает:
        Среднее время в миллисекундах
    """
    times = []
    
    # Выбор целевого устройства
    target_tensors = tensors[device_type]
    
    for shape, tensor in target_tensors.items():
        # Создание второго тензора для бинарных операций
        b = tensor.clone()
        
        # Прогрев (1 выполнение для компиляции)
        _ = operation(tensor, b)
        if device_type == "gpu":
            torch.cuda.synchronize()
        
        # Измерение времени
        if device_type == "cpu":
            start_time = time.perf_counter()
            for _ in range(num_repeats):
                _ = operation(tensor, b)
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * 1000 / num_repeats
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(num_repeats):
                _ = operation(tensor, b)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / num_repeats
        
        times.append(elapsed)
    
    return np.mean(times)

def main():
    """Основная функция выполнения измерений и анализа."""
    # Подготовка данных
    tensors = create_tensors()
    
    # Определение операций
    operations = [
        ("Матричное умножение", 
         lambda a, b: torch.matmul(a, b)),
        
        ("Поэлементное сложение", 
         lambda a, b: a + b),
        
        ("Поэлементное умножение", 
         lambda a, b: a * b),
        
        ("Транспонирование", 
         lambda a, _: a.transpose(1, 2).contiguous()),
        
        ("Сумма всех элементов", 
         lambda a, _: torch.sum(a))
    ]
    
    # Результаты измерений
    results = defaultdict(dict)
    
    # Измерение производительности
    for op_name, op_func in operations:
        # CPU измерения
        cpu_time = measure_time(op_func, tensors, "cpu")
        results[op_name]["cpu"] = cpu_time
        
        # GPU измерения (если доступно)
        if device.type == "cuda":
            gpu_time = measure_time(op_func, tensors, "gpu")
            results[op_name]["gpu"] = gpu_time
            results[op_name]["speedup"] = cpu_time / gpu_time
    
    # Вывод результатов
    print("\nРезультаты:")
    print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
    print("-" * 65)
    
    for op_name, data in results.items():
        cpu_time = f"{data['cpu']:.2f}"
        
        if device.type == "cuda":
            gpu_time = f"{data['gpu']:.2f}"
            speedup = f"{data['speedup']:.1f}x"
        else:
            gpu_time = "N/A"
            speedup = "N/A"
        
        print(f"{op_name:<25} | {cpu_time:<10} | {gpu_time:<10} | {speedup:<10}")

if __name__ == "__main__":
    main()
