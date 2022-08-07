import constant
import numpy as np


def gauss_zeidel(B, g):
    n = B.shape[0]
    x = np.zeros(n)
    x[0] = 1
    converge = False
    n_iterations = 0
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(B[i][j] * x_new[j] for j in range(i))
            s2 = sum(B[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (g[i] - s1 - s2)
        converge = constant.calc_vector_norm(x_new, x) <= constant.EPS
        x = x_new
        n_iterations += 1
    print("Итераций всего:", n_iterations)
    return x


if __name__ == "__main__":
    A = np.array(constant.CONST_A)
    b = np.array(constant.CONST_B)

    n = A.shape[0]
    B = np.zeros(shape=(n, n))
    g = np.zeros(n)
    for i in range(n):
        g[i] = b[i] / A[i][i]
        for j in range(n):
            if j != i:
                B[i][j] = A[i][j] / A[i][i]
            else:
                B[i][j] = 0
    x = gauss_zeidel(B, g)
    print("Решение СЛАУ:", x)
    r = constant.get_residual_vector(A, x, b)
    print("Вектор невязки:", r)
    print("Норма вектора невязки:",
          constant.calc_vector_norm(r, np.zeros(r.shape[0])))
