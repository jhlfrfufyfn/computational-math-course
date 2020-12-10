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
    for i in range(a_n):
        p[i] = i

    for step in range(a_n):
        max_in_row = -1000
        max_column = 0
        for j in range(a_n):
            if max_in_row < abs(Aw[step][j]):
                max_in_row = abs(Aw[step][j])
                max_column = j

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
    print("x = \n", x)
    print("det(A) = \n", det)
    r = get_residual_vector(A, x, b)
    print("r = Ax - b = \n", r)

    a_inverse = get_inverse_matrix(A)
    print("A^(-1) = \n", a_inverse)
    print("A*A^(-1) = \n", np.matmul(A, a_inverse))

    print("v(a) = \n", get_cond(A, a_inverse))
