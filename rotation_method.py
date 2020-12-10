import math


def step(A, p, k):
    up = 0
    for i in range(k):
        up += A[i][p] * A[i][p]
    up = math.sqrt(up)

    down = 0
    for i in range(k):
        down += A[i][p]*A[i][p]
    down = math.sqrt(down)

    cos_pk = up/down

    up = A[k][p]
    for i in range()

def rotation_method(A, b):


if __name__ == "__main__":
