from sympy import Symbol, solve

import constant
import numpy as np

if __name__ == "__main__":
    a = np.array(constant.CONST_AEig)
    n = a.shape[0]
    At = a.transpose()
    a = np.dot(At, a)
    f = a
    s = np.identity(n)
    for i in range(n - 1):
        m = np.identity(n)
        m[n - 2 - i][:] = f[n - 1 - i][:]
        f = np.dot(m, f)
        f = np.dot(f, np.linalg.inv(m))
        s = np.dot(s, np.linalg.inv(m))  # находим S
    p = f[0][:]  # выделяем p
    x = Symbol('x')
    Lambda = solve(x ** 5 - p[0] * x ** 4 - p[1] * x ** 3 - p[2] * x ** 2 - p[3] * x - p[4], x)
    for l in Lambda:
        y = [l ** i for i in range(n - 1, -1, -1)]
        x = np.dot(s, y)
        print("СВ: ", x)
        r = np.dot(a, x) - l * x
        rnorm = np.linalg.norm(r, 1)
        print("невязка:", rnorm)
