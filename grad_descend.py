import numpy as np
import constant


def grad_descend(A, f):
    n = A.shape[0]
    x = np.zeros(n)
    r = np.dot(A, x) - f
    par = np.dot(r, r) / np.dot(np.dot(A, r), r)
    n_iterations = 0
    converge = False
    while not converge:
        new_x = x - par * r
        r = np.dot(A, new_x) - f
        par = np.dot(r, r) / np.dot(np.dot(A, r), r)
        converge = constant.calc_vector_norm(new_x, x) <= constant.EPS
        x = new_x
        n_iterations += 1
    print("iterations:", n_iterations)
    print("x =", x)
    print("r =", r)
    print("||r|| =", constant.calc_vector_norm(r, np.zeros(r.shape[0])))


if __name__ == "__main__":
    A = np.array(constant.CONST_A)
    b = np.array(constant.CONST_B)

    grad_descend(A, b)
