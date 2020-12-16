import math
import numpy as np
import constant
import math
from constant import CONST_A


def rotation_method(A, b):
    n = A.shape[0]
    for p in range(n):
        for k in range(p + 1, n):
            c = A[p][p] / math.sqrt(A[p][p] * A[p][p] + A[k][p] * A[k][p])
            s = A[k][p] / math.sqrt(A[p][p] * A[p][p] + A[k][p] * A[k][p])

            pth_row = np.copy(A[p])
            kth_row = np.copy(A[k])
            b_copy = np.copy(b)
            for j in range(p, n):
                A[p][j] = c * pth_row[j] + s * kth_row[j]
                b[p] = c * b_copy[p] + s * b_copy[k]

                A[k][j] = -s * pth_row[j] + c * kth_row[j]
                b[k] = -s * b_copy[p] + c * b_copy[k]


if __name__ == "__main__":
    a_matr = np.array(constant.CONST_A,dtype=np.float64)
    b_vect = np.array(constant.CONST_B)
    rotation_method(a_matr, b_vect)
    constant.divide_by_diagonal(a_matr, b_vect)
    x = constant.backward(a_matr, b_vect)
    print("x = ", x)
    print("r = Ax - b = ", constant.get_residual_vector(constant.CONST_A, x, constant.CONST_B))
