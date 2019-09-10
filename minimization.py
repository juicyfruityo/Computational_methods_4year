import numpy as np


def gradient_descent(x_0, eps, method='tangent'):
    """
    x_0 - np.array, начальнео приближение.
    eps - точность пиближения.
    return - np.array, массив со значениями,
    в которых выполнилось условие остановки.
    На каждом шаге происходит движение на один
    шаг градиентного спуска, коэфициент считается
    методом касательных. После каждого шага
    проверяется условие остановки. При выполнении
    условия поиск прекращается.

    """
    delta = np.sqrt(eps)
    x_old = x_0 - 1000*delta
    x_t = x_0  # чтобы условие остановки не выполнилось сразу

    # i = 0
    while not stop_condition(x_old, x_t, delta):
        grad = func_gradient(x_t)
        alpha = evaluate_alpha(method, delta, x_t, grad)
        x_old = x_t
        x_t = x_t - alpha * grad

        # i += 1
        # if i % 100 == 0:
        #     print(x_t, grad, alpha)

    return x_t


def newton_method(x_0, eps, method='fission'):
    """
    x_0 - np.array, начальнео приближение.
    eps - точность пиближения.
    return - np.array, массив со значениями,
    в которых выполнилось условие остановки.
    На каждом шаге происходит движение на один
    шаг согласно методу ньютона, коэфициент считается
    методом дробления шага. После каждого шага
    проверяется условие остановки. При выполнении
    условия поиск прекращается.

    """
    delta = eps
    x_old = x_0 - 1000000*delta
    x_t = x_0  # чтобы условие остановки не выполнилось сразу

    i = 0
    while not stop_condition(x_old, x_t, delta):
        grad = func_gradient(x_t)
        alpha = evaluate_alpha(method, delta, x_t, grad)
        func_2diff_inv = func_2diff_matrix_invert(x_t)
        x_old = x_t
        x_t = x_t - alpha * (func_2diff_inv @ grad)

    return x_t


def evaluate_alpha(method, delta, x_t, grad):
    """
    method='tangent' - методом касательных.
    method='fission' - методом дробления.
    Происходит подсчет коэфицента для шага минимизации.

    """
    if method == 'tangent':
        return evaluate_alpha_tangent(delta, x_t, grad)
    elif method == 'fission':
        return evaluate_alpha_fission(delta, x_t, grad)
    else:
        return 0.005


def evaluate_alpha_tangent(delta, x_t, grad):
    """
    Считается коэфициент с помощью метода касательных
    значение ищется на отрезке [-10, 10] с точностью delta.

    """
    a_n = np.float64(-10)
    b_n = np.float64(10.0)
    c_n = np.float64(0)
    # NEW - add comparsion with old c_n.
    c_n_old = c_n
    x_t = np.float64(x_t)
    grad = np.float64(grad)
    delta =np.float64(delta)
    tao = 0.0001

    # Производная по параметру alpha.
    func_alpha_diff = lambda x: np.float64((func(x_t - (x+tao) * grad) \
                            - func(x_t - (x-tao) * grad)) / (2*tao))
    func_alpha = lambda x: np.float64(func(x_t - x*grad))

    i = 0
    while np.abs(func_alpha_diff(c_n)) > delta:
        # NEW
        c_n_old = c_n
        c_n = (func_alpha(a_n) - func_alpha(b_n) \
            + func_alpha_diff(b_n) * b_n - func_alpha_diff(a_n) * a_n) \
            / (func_alpha_diff(b_n) - func_alpha_diff(a_n))

        if np.abs(func_alpha_diff(a_n)) <= delta:
            return a_n
        elif np.abs(func_alpha_diff(b_n)) <= delta:
            return b_n
        # NEW: костыли;))))
        elif np.abs(a_n - c_n) <= delta:
            return c_n
        elif np.abs(func_alpha_diff(c_n) - func_alpha_diff(c_n_old)) <= delta:
            return c_n

        # i += 1
        # if i <= 100:
        #     print(a_n, b_n, c_n, func_alpha_diff(c_n))

        if func_alpha_diff(c_n) < 0:
            a_n = c_n
        elif func_alpha_diff(c_n) > 0:
            b_n = c_n

    return c_n


def evaluate_alpha_fission(delta, x_t, grad):
    """
    Считается коэфициент с помощью метода
    дробления шага.

    """
    lambd = 0.8
    mu = 2
    beta = 1
    alpha = beta

    func_alpha = lambda x: func(x_t - x*grad)

    while True:
        if func_alpha(alpha) < func_alpha(0):
            return alpha
        else:
            alpha = lambd * alpha



def stop_condition(x_old, x_new, delta):
    """
    Проверятся условие остановки. В случае
    выполнения одного из условий возвращается True.

    """
    x_dist = np.abs(np.linalg.norm(x_old - x_new))
    f_dist = np.abs(np.linalg.norm(func(x_old)
                                    - func(x_new)))
    grad = np.abs(np.linalg.norm(func_gradient(x_new)))

    if x_dist <= delta:
        return True
    elif f_dist <= delta:
        return True
    elif grad <= delta:
        return True
    # if x_dist <= delta and f_dist <= delta and \
    #     grad <= delta:
    #     return True
    else:
        return False


def func_2diff_matrix_invert(x):
    """
    Возвращает обратную матрицу к матрице
    вторых производных функции в точке x.

    """
    func_2diff_matr = func_2diff_matrix(x)
    func_2diff_inv = func_matrix_invert(func_2diff_matr)
    return func_2diff_inv


def func_2diff_matrix(x):
    """
    Считается матрица вторых производных в точке x.

    """
    matrix = np.zeros((x.shape[0], x.shape[0]))
    tao = 0.0001

    # Заполняются вначале диагональные элементы матрицы.
    for i in range(x.shape[0]):
        x_i_plustao = x.astype(np.float64)
        x_i_plustao[i] += tao
        x_i_minustao = x.astype(np.float64)
        x_i_minustao[i] -= tao

        matrix[i, i] = (func(x_i_minustao) - 2*func(x)
                        + func(x_i_plustao)) / tao**2

    # Заполняю элементы соответствующие смешанным производным.
    # Предполагается d2f/dxdy == d2f/dydx.
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            if i == j:
                continue

            x_i_plustao = x.astype(np.float64)
            x_i_plustao[i] += tao
            x_i_minustao = x.astype(np.float64)
            x_i_minustao[i] -= tao

            x_j_plustao = x.astype(np.float64)
            x_j_plustao[j] += tao
            x_j_minustao = x.astype(np.float64)
            x_j_minustao[j] -= tao

            x_ij_plusminustao = x.astype(np.float64)
            x_ij_plusminustao[i] += tao
            x_ij_plusminustao[j] -= tao
            x_ji_plusminustao = x.astype(np.float64)
            x_ji_plusminustao[j] += tao
            x_ji_plusminustao[i] -= tao

            f_ij = (func(x_i_plustao) - func(x)
                    - func(x_ij_plusminustao) + func(x_j_minustao)) / tao**2
            f_ji = (func(x_j_plustao) - func(x)
                    - func(x_ji_plusminustao) + func(x_i_minustao)) / tao**2

            matrix[i, j] = (f_ij + f_ji) / 2.0
            matrix[j, i] = (f_ij + f_ji) / 2.0  # Симметрия.

    return matrix


def func_matrix_invert(matrix):
    """
    Считается обратная матрица к данной.

    """
    return np.linalg.inv(matrix)


def func_gradient(x):
    """
    x - np.array, вектор со значениями.
    Возвращает значение первой производной
    функции в данной точке.

    """
    tao = 0.0001
    grad = []

    for i, _ in enumerate(x):
        x_i_plustao = x.astype(np.float64)
        x_i_plustao[i] += tao
        x_i_minustao = x.astype(np.float64)
        x_i_minustao[i] -= tao

        grad_i = (func(x_i_plustao) - func(x_i_minustao)) / (2*tao)
        grad.append(grad_i)

    return np.array(grad, dtype=np.float64)


def func(x):
    """
    x - np.array, вектор со значениями.
    Возвращается значение функции в данной точке.

    """
    # Всё  работает кроме невыпуклых, и иногда если
    # точка очень далеко от минимума, то морось.
    # f = (x[0]-1)**2 + 100*x[1]**2 + 0.3*x[2]**2 + 5 \
    #      + (x[0]-1)**2 * x[1]**2
    # f = x[0]**4 + x[1]**4 - (x[0] + x[1])**2
    # f =  x[0]**2 - x[1]**2
    # f = np.float64(x[0]**2 + x[1]**4)
    # f = (x[0] - 1)**2 + (x[1] + 1)**2
    # f = 2 * x[0]**2 + x[0] * x[1] + 2 * x[1]**2
    # f = 2 * x[0]**3 - x[0] * x[1] + 2 * x[1]**3
    # f = x[0]**2 - x[0]*x[1] + x[1]**2 - 2*x[0] + x[1]
    # f = 2 * x[0]**2 + x[0] * x[1] + 2 * x[1]**2
    # f = (1 - x[0])**2 + (x[1] - x[0]**2)**2
    # f = x[0]**2 + 5 * x[1]**2 + 3 * x[2]**2 + 4 * x[0]*x[1] \
    #     - 2 * x[1]*x[2] - 2 * x[0]*x[2]
    return np.float64(f)


if __name__ == '__main__':
    # TESTING
    x = np.array([3, 4, 5], dtype=np.float64)
    # print(func(x), func_gradient(x))
    # print(func_2diff_matrix(x))
    # print(func_2diff_matrix_invert(x))
    # print(evaluate_alpha_tangent(0.0001, x, func_gradient(x)))

    # Тут следует задать точность минимизации
    # и начальное приближение.
    eps = 10**(-8)
    x_0 = np.array([10, -30])

    # Первый этап - приближение с помощью градиентного
    # спуска, с подбором коэфициента методом касательных.
    # Точность приближения sqrt(eps).
    x_min = gradient_descent(x_0, eps)
    print(x_min)

    # Второй этап - приближение методом ньютона
    # с подбором парметра шага методом дробления.
    # Точность приближения eps
    x_min = newton_method(x_min, eps)

    print("Result = ", end='')
    print(x_min)
