import numpy as np
import math
import matplotlib.pyplot as plt
import time
import gc


def integral_u_diff(a, b, eps, N, uPrev, uNext, solution=False):
    '''

    '''
    # h1 = (b - a) / float(N / 2)  # h for uPrev
    # h2 = (b - a) / float(N)  # h for uNext
    # uPrev_func = lambda x: f_func(x) \
    #     + h1 * (uPrev @ np.fromfunction(lambda i: K_func(x, a + i*h1), (int(N/2), )))
    # uNext_func = lambda x: f_func(x) \
    #     + h2 * (uNext @ np.fromfunction(lambda i: K_func(x, a + i*h2), (N, )))
    #
    # integrate_func = lambda x: (uNext_func(x) - uPrev_func(x))**2
    if solution == False:
        integrate_func = lambda x: (u_func (x, uNext, a, b)
                                    - u_func (x, uPrev, a, b))**2
    else:
        integrate_func = lambda x: (solution_func(x) - u_func (x, uPrev, a, b))**2
    # function = lambda x: (uNext_func(x) - uPrev_func(x))**2
    # integrate_func = lambda x: x**2 + 2

    # Дальше идет непосредственно сам метод интегрирования слева-направо.
    h = (b - a)
    integral_result = 0.0
    p = 2
    epsA = eps / (b - a)
    eps0 = eps

    while a < b:
        Ih = 0
        Ihdiv2 = 0
        delta = 10.0

        while delta > eps0 * Ihdiv2 and delta > epsA * h:
            h /= 2

            Ih = 0.5 * (integrate_func(a) + integrate_func(a + h)) * h
            Ihdiv2 = 0.5 * (integrate_func(a)
                            + integrate_func(a + h/2)) * (h/2) \
                     + 0.5 * (integrate_func(a + h/2)
                              + integrate_func(a + h)) * (h/2)

            delta = (Ihdiv2 - Ih) / (2**p - 1)
        # print(Ih, a, h, delta)
        # break
        integral_result += Ih
        # print(a, b, h, delta)
        # print(integral_result)
        h *= 2
        a += h

        delta = delta**p
        if delta > epsA and delta > epsO*Ihd2:
            h *= 2

        if a + h > b:
            h = b - a
        # else:
        #     h *= 2

    return integral_result
    # return 10**(-20)


def solve_linear_system(a, b, N):
    '''
    a, b - начало и конец отрезка.
    N - количество точек разбиения.
    retutn - вектор U который является решением
    системы на основе составной квадратурной формулы
    (в данном случае - формулы Симпсона).

    '''
    h = (b - a) / N

    # С использованием формулы Симпсона.
    K = np.fromfunction(lambda i, j: h/6
            * (K_func(a + i*h, a + j*h)
                + K_func(a + (i+1)*h, a + (j+1)*h)
                + 4*K_func(a + (i+1.0/2.0)*h, a + (j+1.0/2.0)*h)),
               (N, N))
    f = np.fromfunction(lambda i: f_func(a + i*h), (N,))

    # Solving (I - K)U = f
    # print(f)
    # print(f.shape, K.shape)
    U = np.linalg.inv(np.eye(N, N) - K) @ f
    # U = np.matmul(np.linalg.inv(np.eye(N) - K), f)

    return U


def u_func (x, u, a, b):

    h = (b - a)/u.shape [0]
    return f_func (x) \
     + h*np.dot (u, np.fromfunction (lambda i: K_func(x, a + h/2 + i*h),
                                     (u.shape [0], )))


def K_func(x, t):
    return 1.0 / 2.0 * x * np.exp(t)  # 1
    # return np.sin(x * t)  # 2
    # return np.sin(x) * np.cos(t)  # 3
    # return x * t**2  #


def f_func(x):
    return np.exp(-x)  # 1
    # return 1 + 1 / x * (np.cos(x/2) - 1)  # 2
    # return np.cos(2*x)  # 3
    # return 1 + 0*x  #


def solution_func(x):
    return x + np.exp(-x)  # 1
    # return 1  # 2
    # return np.cos(2*x)  # 3
    # return 1 + 4.0/9.0 * x  #


if __name__ == '__main__':
    a, b = 0.0, 1.0  # 1, 4
    # a, b = 10**(-6), 0.5  # 2
    # a, b = 0, 2*np.pi  # 3
    eps = 10**(-4)
    N = 8
    delta = 10

    # thrl = 100
    uPrev = solve_linear_system(a, b, N)
    # uPrev = get_approximation_values(a, b, N)
    uNext = uPrev
    # print(uPrev[[i for i in range(0, N*thrl, thrl)]])
    # print(np.array([solution_func(a+i*(b-a)/N) for i in range(N)]))

    # st_solve = time.time()
    # uNext = solve_linear_system(a, b, N*512)
    # et_solve = time.time()
    # print("Истекшее время на solve system = {0}".format(et_solve - st_solve))

    while delta > eps:
        N *= 2
        uPrev = uNext
        st_solve = time.time()
        uNext = solve_linear_system(a, b, N)
        # uNext = get_approximation_values(a, b, N)
        et_solve = time.time()
        print("Истекшее время на solve system = {0}".format(et_solve - st_solve))

        gc.collect()
        # TODO: посчитать норму отклонения решеий - это будет delta.
        # delta = np.sqrt(integral_u_diff(a, b, eps, N, uPrev, uNext))
        delta = np.sqrt(np.abs(integral_u_diff(a, b, eps, N, uPrev, uNext)))
        # delta = np.sqrt (integrate (a, b, eps, eps))
        # delta = np.sqrt (integrate (a, b, eps, eps,
        #     lambda x: (u_func (x, uNext, a, b) - u_func (x, uPrev, a, b))**2))
        print("Delta = {0} количество разбиений = {1}".format(delta, N))

        if (N == 8192):
            break

    # print(np.sqrt(integral_u_diff(a, b, eps, N, uPrev, uNext)))

    solution = np.array([solution_func(a+i*(b-a)/N) for i in range(N)])
    # diff = np.linalg.norm(solution - uNext)
    diff = np.sqrt(np.abs(integral_u_diff(a, b, eps, N, uNext, solution, True)))
    print("\nL2 difference between actualn and program solutions = {0}".format(diff))
    diff = np.max(np.abs(solution - uNext))
    print("C difference between actualn and program solutions = {0}".format(diff))
    # TODO: сделать рисунок решения реального и приближенного.
    # А также сравнения его с точным решением.
    plt.plot([i for i in range(N)], solution)
    plt.plot([i for i in range(N)], uNext)
    plt.show()
