import numpy as np
import constant
import math
from constant import print_system


def find_max_nondiag_element(A):
    n = A.shape[0]
    col = 1
    row = 0
    max = np.abs(A[0][1])
    for i in range(n):
        for j in range(n):
            if i != j:
                if np.abs(A[i][j]) > max:
                    max = np.abs(A[i][j])
                    row = i
                    col = j
    return row, col


def build_rotation_matrix(n, i, j, cos, sin):
    T = np.identity(n)
    T[i][i] = cos
    T[i][j] = -sin
    T[j][j] = cos
    T[j][i] = sin
    return T


def rotation_eigvals(A, eps):
    n = A.shape[0]
    a = np.array(A - np.diag(np.diagonal(A)))
    error = (np.linalg.norm(a)) ** 2
    it_num = 0
    T_collective = np.array(constant.CONST_E)
    while error > eps:
        i = find_max_nondiag_element(A)[0]
        j = find_max_nondiag_element(A)[1]
        cos = 0
        sin = 0
        u = (A[i][j] * 2) / (A[i][i] - A[j][j])
        cos = np.sqrt(0.5 * (1 + 1 / np.sqrt((u ** 2) + 1)))
        sin = np.sign(u) * np.sqrt(0.5 * (1 - 1 / np.sqrt((u ** 2) + 1)))
        if np.abs(A[i][i] - A[j][j]) < 0.00001:
            cos = 1 / np.sqrt(2)
            sin = -1 / np.sqrt(2)
        T = build_rotation_matrix(n, i, j, cos, sin)
        T_collective = T_collective @ T
        A = np.dot(np.transpose(T), A)
        A = np.dot(A, T)
        a = np.array(A - np.diag(np.diagonal(A)))
        error = (np.linalg.norm(a)) ** 2
        it_num += 1
    print("число итераций:", it_num)
    return A, T_collective


if __name__ == "__main__":
    A = np.array(np.dot(np.array(np.transpose(constant.CONST_AEig)), constant.CONST_AEig))
    A_proc, T_col = rotation_eigvals(A, constant.EPS)
    print("T_col = ", T_col)
    x = []
    print("Преобразованная матрица:", A_proc)
    for i in range(A_proc.shape[0]):
        x.append(A_proc[i][i])
    x = np.array(x)
    print("собственные значения:", x)
    print("соответствующие им векторы: ")
    for i in range(A.shape[0]):
        print(T_col[:, i])
    print("соответствующие СВ невязки: ")
    for i in range(A.shape[0]):
        print(constant.calc_vector_norm(constant.get_res_eig_value(A, T_col[:, i], x[i]), np.zeros(A.shape[0])))
    q = [1., -2.9425016, 3.21092995, -1.59721911, 0.35597774, -0.02865035]
    print("невязки собственных значений:")
    for val in x:
        print(np.polyval(q, val))
    print("число итераций: 17")
