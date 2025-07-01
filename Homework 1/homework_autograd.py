import torch


def simple_gradient_computation(x_val, y_val, z_val):
    """
    Вычисляет функцию f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z и её градиенты.

    Параметры:
        x_val, y_val, z_val (float): Значения для переменных x, y, z.

    Возвращает:
        tuple: (значение функции, градиенты по x, y, z)
    """
    # Проверка типов входных данных
    if not all(isinstance(v, (int, float)) for v in (x_val, y_val, z_val)):
        raise TypeError("Все аргументы должны быть числами (int или float)")

    # Создаем тензоры с возможностью вычисления градиентов
    x = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float32, requires_grad=True)
    z = torch.tensor(z_val, dtype=torch.float32, requires_grad=True)

    # Вычисляем функцию
    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z

    # Вычисляем градиенты
    f.backward()

    # Проверка аналитического решения
    df_dx_analytical = 2 * x_val + 2 * y_val * z_val
    df_dy_analytical = 2 * y_val + 2 * x_val * z_val
    df_dz_analytical = 2 * z_val + 2 * x_val * y_val

    # Сравнение с автоматическими градиентами с обработкой возможных ошибок
    try:
        assert torch.allclose(x.grad, torch.tensor(df_dx_analytical)), "Неверный градиент по x"
        assert torch.allclose(y.grad, torch.tensor(df_dy_analytical)), "Неверный градиент по y"
        assert torch.allclose(z.grad, torch.tensor(df_dz_analytical)), "Неверный градиент по z"
    except AssertionError as e:
        print(f"Ошибка проверки градиентов: {e}")
        raise

    return f.item(), x.grad.item(), y.grad.item(), z.grad.item()


def mse_gradient(y_true, y_pred, w=None, b=None):
    """
    Вычисляет MSE и градиенты по параметрам линейной модели y_pred = w * x + b.

    Параметры:
        y_true (torch.Tensor): Истинные значения, shape (n,)
        y_pred (torch.Tensor): Предсказанные значения, shape (n,)
        w (torch.Tensor, optional): Параметр весов с requires_grad=True
        b (torch.Tensor, optional): Параметр смещения с requires_grad=True

    Возвращает:
        tuple: (значение MSE, градиент по w, градиент по b)
    """
    # Проверка размерностей
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую размерность")

    # Проверка, что y_pred требует градиенты
    if not y_pred.requires_grad:
        raise RuntimeError("y_pred должен быть создан с requires_grad=True")

    # Вычисляем MSE
    loss = torch.mean((y_pred - y_true) ** 2)

    # Вычисляем градиенты
    loss.backward()

    # Получаем градиенты, если параметры переданы
    grad_w = w.grad.item() if w is not None else None
    grad_b = b.grad.item() if b is not None else None

    return loss.item(), grad_w, grad_b


def chain_rule_example(x_val):
    """
    Вычисляет f(x) = sin(x^2 + 1) и её градиент df/dx.

    Параметры:
        x_val (float): Входное значение.

    Возвращает:
        tuple: (значение функции, градиент)
    """
    if not isinstance(x_val, (int, float)):
        raise TypeError("x_val должен быть числом (int или float)")

    # Первый способ: backward()
    x1 = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    f1 = torch.sin(x1 ** 2 + 1)
    f1.backward()
    grad1 = x1.grad.item()

    # Второй способ: autograd.grad() (на новых тензорах)
    x2 = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    f2 = torch.sin(x2 ** 2 + 1)
    grad2 = torch.autograd.grad(f2, x2)[0].item()

    # Проверка совпадения результатов с обработкой ошибок
    try:
        assert torch.allclose(torch.tensor(grad1), torch.tensor(grad2), atol=1e-6), "Градиенты не совпадают"
    except AssertionError as e:
        print(f"Ошибка сравнения градиентов: {e}")
        raise

    # Аналитическая проверка
    analytical_grad = 2 * x_val * torch.cos(torch.tensor(x_val ** 2 + 1)).item()
    try:
        assert torch.allclose(torch.tensor(grad1), torch.tensor(analytical_grad), rtol=1e-4), "Неверный градиент"
    except AssertionError as e:
        print(f"Ошибка аналитической проверки: {e}")
        raise

    return f1.item(), grad1


def test_simple_gradient():
    """Тест для простых вычислений с градиентами."""
    print("\n2.1 Простые вычисления с градиентами")
    try:
        x, y, z = 1.0, 2.0, 3.0
        f, df_dx, df_dy, df_dz = simple_gradient_computation(x, y, z)

        print(f"f({x}, {y}, {z}) = {f}")
        print(f"df/dx = {df_dx}, df/dy = {df_dy}, df/dz = {df_dz}")
        print("\n============")

        expected_f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
        if not abs(f - expected_f) < 1e-6:
            raise AssertionError("Неверное значение функции")
    except Exception as e:
        print(f"Тест не пройден: {e}")
        raise


def test_mse_gradient():
    """Тест для градиента MSE."""
    print("\n2.2 Градиент функции потерь")
    try:
        # Создаем данные
        n = 5
        x = torch.linspace(0, 1, n)
        w = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)

        # Линейная модель
        y_pred = w * x + b
        y_true = torch.tensor([1.1, 1.9, 3.2, 3.8, 5.1])

        # Вычисляем MSE и градиенты
        loss, grad_w, grad_b = mse_gradient(y_true, y_pred, w, b)

        print(f"MSE = {loss}")
        print(f"grad w = {grad_w}, grad b = {grad_b}")
        print("\n============")

    except Exception as e:
        print(f"Тест не пройден: {e}")
        raise


def test_chain_rule():
    """Тест для цепного правила."""
    print("\n2.3 Цепное правило")
    try:
        x_val = 0.5
        f, df_dx = chain_rule_example(x_val)

        print(f"f({x_val}) = sin({x_val}^2 + 1) = {f}")
        print(f"df/dx = {df_dx}")

    except Exception as e:
        print(f"Тест не пройден: {e}")
        raise


if __name__ == "__main__":
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Запуск тестов
    try:
        test_simple_gradient()
        test_mse_gradient()
        test_chain_rule()
        print("\n=== Все тесты завершены ===")
        
    except Exception as e:
        print(f"\nОшибка при выполнении тестов: {e}")
