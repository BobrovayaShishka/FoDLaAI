import torch


def create_tensors():
    """
    Создает и возвращает различные типы тензоров с демонстрационными значениями.

    Возвращает:
        tuple: (tensor_random, tensor_zeros, tensor_ones, tensor_range)
    """
    try:
        # 1.1 Создание тензоров
        tensor_random = torch.rand(3, 4)
        tensor_zeros = torch.zeros(2, 3, 4)
        tensor_ones = torch.ones(5, 5)
        tensor_range = torch.arange(16).reshape(4, 4)

        return tensor_random, tensor_zeros, tensor_ones, tensor_range
    except Exception as e:
        raise RuntimeError(f"Ошибка при создании тензоров: {e}")


def tensor_operations(A, B):
    """
    Выполняет основные операции с тензорами.

    Параметры:
        A (torch.Tensor): Матрица 3x4
        B (torch.Tensor): Матрица 4x3

    Возвращает:
        tuple: (A_transposed, matmul, elementwise, sum_A)
    """
    if A.shape != (3, 4) or B.shape != (4, 3):
        raise ValueError("Некорректные размеры входных матриц")

    try:
        A_transposed = A.T
        matmul = torch.matmul(A, B)
        elementwise = A * B.T
        sum_A = A.sum()

        return A_transposed, matmul, elementwise, sum_A
    except Exception as e:
        raise RuntimeError(f"Ошибка при выполнении операций: {e}")


def tensor_indexing(tensor):
    """
    Демонстрирует различные методы индексации тензоров.

    Параметры:
        tensor (torch.Tensor): Исходный тензор 5x5x5

    Возвращает:
        tuple: (first_row, last_col, center_submatrix, even_indices)
    """
    if tensor.dim() != 3 or tensor.shape != (5, 5, 5):
        raise ValueError("Ожидается тензор размерности 5x5x5")

    try:
        first_row = tensor[0, :, :]
        last_col = tensor[:, :, -1]
        center_submatrix = tensor[2:4, 2:4, 2]
        even_indices = tensor[::2, ::2, ::2]

        return first_row, last_col, center_submatrix, even_indices
    except Exception as e:
        raise RuntimeError(f"Ошибка при индексации: {e}")


def tensor_reshaping(tensor):
    """
    Демонстрирует изменение формы тензора.

    Параметры:
        tensor (torch.Tensor): Исходный одномерный тензор

    Возвращает:
        list: Список тензоров с разными формами
    """
    if tensor.dim() != 1:
        raise ValueError("Ожидается одномерный тензор")

    try:
        shapes = [
            (2, 12),
            (3, 8),
            (4, 6),
            (2, 3, 4),
            (2, 2, 2, 3)
        ]
        return [tensor.reshape(shape) for shape in shapes]
    except Exception as e:
        raise RuntimeError(f"Ошибка при изменении формы: {e}")


def test_tensor_operations():
    """Тестирование операций с тензорами"""
    print("\n=== Тестирование операций с тензорами ===")

    # Тест создания тензоров
    try:
        tensors = create_tensors()
        print("\n1.1 Создание тензоров")
        for i, tensor in enumerate(tensors):
            print(f"Тензор {i + 1}:\n{tensor}")
        print("\n============")

    except Exception as e:
        print(f"Ошибка в create_tensors(): {e}")

    # Тест операций
    try:
        A = torch.rand(3, 4)
        B = torch.rand(4, 3)
        results = tensor_operations(A, B)
        print("\n1.2 Операции с тензорами")
        print("A:\n", A)
        print("B:\n", B)
        print("Транспонированная A:\n", results[0])
        print("Матричное умножение:\n", results[1])
        print("Поэлементное умножение:\n", results[2])
        print("Сумма элементов A:", results[3])
        print("\n============")

    except Exception as e:
        print(f"Ошибка в tensor_operations(): {e}")

    # Тест индексации
    try:
        tensor = torch.rand(5, 5, 5)
        idx_results = tensor_indexing(tensor)
        print("\n1.3 Индексация и срезы")
        print("Первая строка:\n", idx_results[0])
        print("Последний столбец:\n", idx_results[1])
        print("Центральная подматрица:\n", idx_results[2])
        print("Элементы с четными индексами:\n", idx_results[3])
        print("\n============")

    except Exception as e:
        print(f"Ошибка в tensor_indexing(): {e}")

    # Тест изменения формы
    try:
        tensor = torch.arange(24)
        reshapes = tensor_reshaping(tensor)
        print("\n1.4 Работа с формами")
        for i, reshaped in enumerate(reshapes):
            print(f"Форма {i + 1}:\n{reshaped}")

    except Exception as e:
        print(f"Ошибка в tensor_reshaping(): {e}")


if __name__ == "__main__":
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Запуск тестов
    test_tensor_operations()

    print("\n=== Все тесты завершены ===")
