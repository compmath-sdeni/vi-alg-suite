import numpy as np
from scipy import linalg, vectorize
import random
from tools.print_utils import vectorToString

from constraints.allspace import Rn
from problems.matrix_oper_vi import MatrixOperVI


def getProblem(M: int, withB: bool, printMatr: bool = False):
    def elfun(i, j):
        if (j + 1 == M - i) and j > i:
            return -1
        elif (j + 1 == M - i) and j < i:
            return 1
            # elif (j == i):
            # return -1
        else:
            return 0

    # A = np.zeros((M, M), dtype=int)
    # A = np.identity(M, dtype=int)
    # A = np.fromfunction(vectorize(lambda i,j: random.randint(0,10)), (M,M), dtype=float)

    A = np.fromfunction(vectorize(lambda i, j: elfun(i, j)), (M, M), dtype=float)

    if printMatr:
        print("A:\n", A)

    if withB:
        x = np.fromfunction(vectorize(lambda i: i), (M,), dtype=float)
        print("Desired x: ", vectorToString(x))
        b = A @ x
        print("b: ", vectorToString(b))
    else:
        b = np.zeros((M,), dtype=int)

    return MatrixOperVI(A=A, b=b, x0=Rn(M).getSomeInteriorPoint())
