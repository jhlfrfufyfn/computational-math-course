import numpy as np
import constant
from constant import swap_columns
from constant import swap_values
from constant import get_norm
from constant import get_residual_vector


def gauss_method(A, b):
    a_det = 1

    a_n, a_cols = A.shape
    Aw = np.copy(A)
    Bw = np.copy(b)
    p = np.array(np.zeros(a_n), dtype=np.int8)
    count_perm = 0
    for i in range(a_n):
        p[i] = i

    for step in range(a_n):
        max_in_row = -1000
        max_column = 0
        for j in range(a_n):
            if max_in_row < abs(Aw[step][j]):
                max_in_row = abs(Aw[step][j])
                max_column = j

        if step == max_column:
            count_perm+=1
        swap_columns(Aw, step, max_column)
        p[step], p[max_column] = swap_values(p[step], p[max_column])

        pivot = Aw[step][step]

        a_det *= pivot

        for j in range(a_n):
            Aw[step][j] /= pivot
        Bw[step] /= pivot
        Aprev = np.copy(Aw)
        for j in range(step + 1, a_n):
            for c in range(step, a_n):
                Aw[j][c] -= Aprev[step][c] * Aprev[j][step]
            Bw[j] -= Bw[step] * Aprev[j][step]

    x = constant.backward(Aw, Bw)

    newx = np.array(np.zeros(a_n))
    for i in range(a_n):
        newx[p[i]] = x[i]

    a_det = a_det*((-1)**count_perm)
    return newx, a_det


def test_gauss():
    matrix = np.array([
        [4, 2, -2]
        , [2, 1, 2]
        , [8, 6, 6]
    ], dtype=np.float64)

    b = np.zeros(3)
    for i in range(3):
        b[i] = i

    print(gauss_method(matrix, b))
    print(np.linalg.solve(matrix, b))


def get_inverse_matrix(A):
    a_n, a_cols = A.shape

    E = np.zeros(shape=(a_n, a_n))
    for i in range(a_n):
        E[i][i] = 1

    a_inverse = np.zeros(shape=(a_n, a_n))
    for i in range(a_n):
        curr_b = E[:, i]
        x_e = gauss_method(A, curr_b)[0]
        a_inverse[:, i] = x_e

    return a_inverse


def get_cond(A, A_inverse):
    return get_norm(A) * get_norm(A_inverse)


if __name__ == "__main__":
    A = np.array(constant.CONST_A)
    b = np.array(constant.CONST_B)

    (x, det) = gauss_method(A, b)
    print("Решение СЛАУ:", x)
    print("Определитель:", det)
    r = get_residual_vector(A, x, b)
    print("Вектор невязок:", r)
    print("Норма вектора невязок:", constant.calc_vector_norm(r, np.zeros(r.shape[0])))

    a_inverse = get_inverse_matrix(A)
    print("Обратная матрица:", a_inverse)
    print("Норма матрицы A*A^(-1) - Е =", constant.get_norm(np.matmul(A, a_inverse)-constant.CONST_E))

    print("Число обусловленности:", get_cond(A, a_inverse))
