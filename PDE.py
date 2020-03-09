import numpy as np
import math

import matplotlib.pyplot as plt


class ProblemSolver:
    def __init__(self, K11, K12, K21, K22, len1, len2, h1, h2):
        self.K11 = K11
        self.K12 = K12
        self.K21 = K21
        self.K22 = K22
        self.len1 = len1
        self.len2 = len2
        self.h1 = h1
        self.h2 = h2
        self.N1 = int(self.len1 / self.h1)
        self.N2 = int(self.len2 / self.h2)

    def F_function(self, x, y):
        # Правая функция уравнения.
        return 1

    def _solve_mu_k1k2_ij(self, k1, k2):
        mu_k1k2_loc = 0
        for i in range(1, self.N1):
            mu_k2 = 0
            for j in range(1, self.N2):
                mu_k2 += self.F_function(i * self.h1, j * self.h2) \
                         * np.sin(float(k2 * np.pi * j / self.N2))
            mu_k1k2_loc += mu_k2 * np.sin(float(k1 * np.pi * i / self.N1))
        return mu_k1k2_loc

    def _solve_mu_k1k2(self):
        mu_k1k2 = np.zeros((self.N1+1, self.N2+1))
        for k1 in range(1, self.N1):
            for k2 in range(1, self.N2):
                mu_k1k2[k1, k2] = self._solve_mu_k1k2_ij(k1, k2)
        return mu_k1k2

    def _solve_v_ij(self, i, j):
        v = 0
        for k2 in range(1, self.N1):
            v_k2 = 0
            for k1 in range(1, self.N2):
                v_k2 += self.mu_k1k2[k1, k2] * np.sin(float(k1 * np.pi * i / self.N1)) \
                        / (4 * np.sin(float(k1 * np.pi * self.h1 / (2 * self.len1))) ** 2 / self.h1 ** 2
                           + 4 * np.sin(float(k2 * np.pi * self.h2 / (2 * self.len2))) ** 2 / self.h2 ** 2)
            v += v_k2 * np.sin(float(k2 * np.pi * j / self.N2))
        v *= 4 / (self.N1 * self.N2)
        return v

    def poisson_solver(self):
        self.mu_k1k2 = self._solve_mu_k1k2()
        self.v = np.zeros((self.N1+1, self.N2+1))
        for i in range(1, self.N1):
            for j in range(1, self.N2):
                self.v[i, j] = self._solve_v_ij(i, j)
        return self.v


if __name__ == '__main__':
    a = 10
    K11, K12 = 1, a
    K21, K22 = a, 1
    len1, len2 = 10, 10

    # Шаги сетки.
    h1, h2 = 1, 1

    solver = ProblemSolver(K11, K12, K21, K22, len1, len2, h1, h2)

    v = solver.poisson_solver()
    print(v)
