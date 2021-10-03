import numpy as np

from problems.matrix_oper_vi import MatrixOperVI


def getSLE(N: int, *, A: np.ndarray = None, x_test: np.ndarray = None):
    if A is None:
        A = np.random.rand(N, N) * 2.
        A = np.around(A @ A.T, 1)
        print("A:\n", A)

    if x_test is None:
        x_test = np.ones(N)

    b = A @ x_test

    return A, b, x_test


def getProblem(N: int, A: np.ndarray = None):
    M, b, x = getSLE(N, A=A)

    print("Desired x: ", x)
    print("b: ", b)

    return MatrixOperVI(A=M, b=b, x0=np.ones((N,)))
