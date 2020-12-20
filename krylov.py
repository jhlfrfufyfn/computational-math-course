import numpy as np
import numpy.polynomial.polynomial as poly
import constant
import gauss_method

from constant import print_poly

from numpy.linalg import matrix_rank

if __name__ == "__main__":
    A = np.array(constant.CONST_AEig)
    A = np.dot(np.transpose(A), A)

    n = A.shape[0]

    c0 = np.zeros(n)
    c0[0] = 1
    c = [c0]

    for i in range(1, n + 1):
        c.append(np.dot(A, c[i - 1]))
        if matrix_rank(c) != len(c):
            break
    print("c = ", c)

    S = np.transpose(c)[:, 0:5]
    y = np.transpose(c)[:, 5:6]
    print("y = ", y)
    for i in range(S.shape[0]):
        if i < n-i-1:
            constant.swap_columns(S, i, n-i-1)
    print("S = ", S)
    q = gauss_method.gauss_method(S, y)[0]
    q = np.insert(q, 0, 1)
    for i in range(1, q.shape[0]):
        q[i] = -q[i]

    print("коэффициенты характеристического многочлена =", q)
    # print_poly(A)
    exact_eigenvalues = np.linalg.eigvals(A)
    vals = poly.polyroots(list(reversed(q)))
    print("невязки собственных значений:")
    for val in exact_eigenvalues:
        print(np.polyval(q, val))
