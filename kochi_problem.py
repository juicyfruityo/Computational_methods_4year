import numpy as np
import matplotlib.pyplot as plt
import math
import time
from abc import ABC, abstractmethod


class AbstractSolver(ABC):
    def __init__(self, start, end, error, condition, function):
        '''
        start - начало отрезка.
        end - конец отрезка.
        error - требуемая точность вычислений.
        function - функция, задающая правую часть уравнения.
            (для неё требуется держать счетчик числа обращений)
        condition - вектор начальных условий для x == start.
        (Я так понимаю x - одномерный всегда)

        '''
        self.start = start
        self.end = end
        self.error = error
        self.condition = condition
        self.function = function

        # Вектор значений k1, k2, .. k5., нумерация для
        # удобства будет с 1.
        self.K = [0 for i in range(6)]
        self.x = [self.start]  # Массив значений x.
        self.y = [self.condition]  # Массив векторных значений y.
        self.y_add = 0
        self.x_add = 0
        self.h = 0.01  # Шаг разбиения.

    @abstractmethod
    def _calcuate_K(self, num):
        '''
        Считаются значения k1, k2, .. k5.
        (офкорс только те, которые нужны)
        num - номер n, для которого нужно посчитать,

        '''
        pass

    @abstractmethod
    def _calculate_y(self, num):
        '''
        Считаются значения y_n+1.
        num - номер n, для которого нужно посчитать,

        '''
        pass

    @abstractmethod
    def _calculate_loss(self, num):
        '''
        Считается погрешность подсчета y_n+1.
        num - номер n, для которого нужно посчитать,

        '''
        pass

    def _solve_one_iteration(self, num):
        '''
        Решается одна итерация задачи, и ищется следующее
        y_n+1 значение, которое нам подходит, исходя из
        погрешности задачи.
        num - номер n, для которого нужно посчитать,

        '''
        self.y.append(0)  # Проинициализировали следующее значение.
        while True:
            if self.x[num] + self.h > self.end:
                self.h = self.end - self.x[num]

            self._calcuate_K(num)
            self.y[num+1] = self._calculate_y(num)
            current_loss = self._calculate_loss(num)

            # print(current_loss, self.x[num], self.y[num], self.h)

            if current_loss <= self.error:
                next_x = self.x[num] + self.h
                self.x.append(next_x)
                self.h *= 2
                break
            else:
                self.h /= 2

    def solve_problem(self):
        num = 0
        while self.x[num] != self.end:
            self._solve_one_iteration(num)
            num += 1


class FirstSolver(AbstractSolver):
    def __init__(self, start, end, error, condition, function):
        super().__init__(start, end, error, condition, function)

    def _calcuate_K(self, num=None, add=False):
        if add is False:
            Y = self.y[num]
            X = self.x[num]
        else:
            Y = self.y_add
            X = self.x_add

        self.K[1] = self.h * self.function(X, Y)

        x = X + 0.5 * self.h
        y = Y + 0.5 * self.K[1]
        self.K[2] = self.h * self.function(x, y)

        x = X + self.h
        y = Y - self.K[1] + self.K[2]
        self.K[3] = self.h * self.function(x, y)

    def _calculate_y(self, num=None, add=False):
        if add is False:
            Y = self.y[num]
        else:
            Y = self.y_add

        y_Nplus1 = Y + (1.0 / 6) * (self.K[1] + 4*self.K[2]
                                              + self.K[3])
        return y_Nplus1

    def _calculate_additional_y(self, num):
        old_h = self.h
        self.h /= 2.0

        self.x_add = self.x[num]
        self.y_add = self.y[num]
        self._calcuate_K(add=True)
        self.y_add = self._calculate_y(add=True)

        self.x_add = self.x[num] + self.h
        self._calcuate_K(add=True)
        self.y_add = self._calculate_y(add=True)

        additional_y_Nplus1 = self.y_add

        self.h = old_h

        return additional_y_Nplus1


    def _calculate_loss(self, num):
        # y_Nplus1 = self._calculate_y(num)
        y_Nplus1 = self.y[num+1]
        additional_y_Nplus1 = self._calculate_additional_y(num)

        s = 2  # Т.к. порядок задается O(h^2).
        loss = (np.array(y_Nplus1) - np.array(additional_y_Nplus1)) / (1 - 2.0 / 2**(s))
        return np.max(loss)


class SecondSolver(FirstSolver):
    def __init__(self, start, end, error, condition, function):
        super().__init__(start, end, error, condition, function)

    def _calcuate_additional_K(self, num):
        self.K[1] = self.h * self.function(self.x[num], self.y[num])

        x = self.x[num] + 0.5 * self.h
        y = self.y[num] + 0.5 * self.K[1]
        self.K[2] = self.h * self.function(x, y)

        x = self.x[num] + 0.5 * self.h
        y = self.y[num] + 0.5 * self.K[2]
        self.K[3] = self.h * self.function(x, y)

        x = self.x[num] + self.h
        y = self.y[num] + self.K[3]
        self.K[4] = self.h * self.function(x, y)

    def _calculate_additional_y(self, num):
        y_Nplus1 = self.y[num] + (1.0 / 6) * (self.K[1] + 2*self.K[2]
                                              + 2*self.K[3] + self.K[4])
        return y_Nplus1

    def _calculate_loss(self, num):
        y_Nplus1 = self.y[num+1]

        self._calcuate_additional_K(num)
        additional_y_Nplus1 = self._calculate_additional_y(num)

        loss = np.max(np.array(y_Nplus1) - np.array(additional_y_Nplus1))
        return loss


class ThirdSolver(AbstractSolver):
    def __init__(self, start, end, error, condition, function):
        super().__init__(start, end, error, condition, function)

    def _calcuate_K(self, num):
        self.K[1] = self.h * self.function(self.x[num], self.y[num])

        x = self.x[num] + 1.0 / 3 * self.h
        y = self.y[num] + 1.0 / 3 * self.K[1]
        self.K[2] = self.h * self.function(x, y)

        x = self.x[num] + 1.0 / 3 * self.h
        y = self.y[num] + 1.0 / 6 * (self.K[1] + self.K[2])
        self.K[3] = self.h * self.function(x, y)

        x = self.x[num] + 0.5 * self.h
        y = self.y[num] + 1.0 / 8 * self.K[1] + 3.0 / 8 * self.K[3]
        self.K[4] = self.h * self.function(x, y)

        x = self.x[num] + self.h
        y = self.y[num] + 0.5 * self.K[1] - 1.5 * self.K[3] + 2 * self.K[4]
        self.K[5] = self.h * self.function(x, y)

    def _calculate_y(self, num):
        y_Nplus1 = self.y[num] + 1.0 / 6 * (self.K[1] + 4*self.K[4]
                                            + self.K[5])
        return y_Nplus1

    def _calculate_loss(self, num):
        loss = 1.0 / 30 * (2*self.K[1] - 9*self.K[3] + 8*self.K[4] - self.K[5])
        return np.max(loss)


def decorator_counter(func):
    def wrapper_func(x, y):
        wrapper_func.count += 1
        res = func(x, y)
        return res
    wrapper_func.count = 0
    return wrapper_func


@decorator_counter
def right_function(x, y):
    # return 1
    # return x
    return np.array([y[1], y[0]])
    # return np.array([y[0]])


if __name__ == '__main__':
    start = 0
    end = 1
    error = 1e-3
    # condition = 0
    condition = np.array([1, 1])
    # condition = np.array([1])
    function = right_function

    # solver = FirstSolver(start, end, error, condition, function)
    # solver = SecondSolver(start, end, error, condition, function)
    solver = ThirdSolver(start, end, error, condition, function)

    time_start = time.time()
    solver.solve_problem()
    time_end = time.time()
    print("Time for solving using method = {0} \
          \nNumber of function calls = {1}"
          .format(time_end - time_start, function.count))

    plt.plot(solver.x, np.array(solver.y)[:, 0])
    # plt.plot(solver.x, solver.y)
    plt.show()
