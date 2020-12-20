import constant
import numpy as np


def pow_method(A):
    n = A.shape[0]

    y = np.ones(n)
    l = np.ones(n)
    converge = False
    iter = 0
    while not converge:
        last_y = np.copy(y)
        last_l = np.copy(l)
        y = np.dot(A, y)
        for i in range(n):
            l[i] = y[i] / last_y[i]
        maxi = -1000
        for i in range(n):
            maxi = max(maxi, np.abs(l[i] - last_l[i]))
        converge = maxi <= constant.EPS
        y /= np.linalg.norm(y)
        iter += 1
    print("итераций: ", iter)
    return np.average(l), y


if __name__ == "__main__":
    A = np.dot(np.transpose(constant.CONST_AEig), np.array(constant.CONST_AEig))
    (l, vect) = pow_method(A)
    print("максимальное собственное значение:", l)
    print("соответствующий собственный вектор:", vect)
    print("невязка собственного вектора:", constant.get_res_eig_value(A, vect, l))
    q = [1., -2.9425016, 3.21092995, -1.59721911, 0.35597774, -0.02865035]
    print("невязка собственного значения: ", np.polyval(q, l))
