import numpy as np
from scipy import linalg, vectorize
import random

from problems.matrix_oper_vi import MatrixOperVI

def getProblem(M:int, A:np.array = None):

    #A = np.zeros((M, M), dtype=int)
    #A = np.identity(M, dtype=int)

    if(A is None):
        A = np.fromfunction(vectorize(lambda i,j: random.randint(0,10)), (M,M), dtype=float)
        A = A@np.transpose(A)

    print("A:\n", A)

    x = np.fromfunction(vectorize(lambda i: i), (M,), dtype=float)
    print("Desired x: ", x)
    #b = np.fromfunction(vectorize(lambda i: random.random()), (M,), dtype=float)
    b = A @ x
    print("b: ", b)

    return MatrixOperVI(A = A, b = b, x0 = np.ones((M,)))
