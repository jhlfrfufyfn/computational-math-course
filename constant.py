import numpy as np

CONST_A = [
    [0.7424, 0.0000, -0.1939, 0.1364, 0.2273]
    , [-0.0455, 0.4848, 0.0000, -0.0924, 0.0303]
    , [0.0152, -0.1364, 0.8787, 0.0167, 0.0530]
    , [0.0455, 0.0000, -0.1106, 0.8787, 0.0000]
    , [0.0303, -0.0455, 0.2197, -0.0182, 0.6363]
]

CONST_B = [3.5330, -3.4254, -2.2483, 1.4120, 2.6634]

CONST_E = [
    [1, 0, 0, 0, 0]
    , [0, 1, 0, 0, 0]
    , [0, 0, 1, 0, 0]
    , [0, 0, 0, 1, 0]
    , [0, 0, 0, 0, 1]
]


def swap_columns(my_array, col1, col2):
    temp = np.copy(my_array[:, col1])
    my_array[:, col1] = my_array[:, col2]
    my_array[:, col2] = temp


def swap_rows(my_array, row1, row2):
    my_array[[row1, row2]] = my_array[[row2, row1]]


def swap_values(a, b):
    return b, a


def get_norm(A):
    n, col_count = A.shape
    max_sum = -1000
    for j in range(n):
        sum = 0
        for i in range(n):
            sum += abs(A[i][j])
        max_sum = max(max_sum, sum)
    return max_sum


def get_residual_vector(A, x, b):
    return np.matmul(A, x) - b


def backward(Aw, Bw):
    a_n, a_cols = Aw.shape
    x = np.array(np.zeros(a_n))
    x[a_n - 1] = Bw[a_n - 1]
    for i in range(a_n - 2, -1, -1):
        x[i] = Bw[i]
        for j in range(i + 1, a_n):
            x[i] -= Aw[i][j] * x[j]
    return x


def divide_by_diagonal(A, b=np.zeros(5)):
    a_n, a_cols = A.shape
    for i in range(a_n):
        pivot = A[i][i]
        for j in range(a_cols):
            A[i][j] /= pivot
        b[i] /= pivot
