from problems.viproblem import VIProblem
import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from constraints.allspace import Rn
from utils.print_utils import vectorToString


# noinspection PyPep8Naming
class MatrixOperVI(VIProblem):
    def __init__(self, A: np.ndarray, b: np.ndarray, C: ConvexSetConstraints = None,
                 x0: np.ndarray = None, hr_name: str= None):
        super().__init__(x0=x0 if x0 is not None else b, hr_name=hr_name)
        self.C = C if C is not None else Rn(A.shape[0])
        self.A = A
        self.b = b

        if (A.shape[0] != x0.shape[0]) or (A.shape[0] != b.shape[0]):
            raise BaseException(
                "Matrix and vector dimentions differs! {0} {1} {2}".format(A.shape[0], x0.shape, b.shape))

    def f(self, x: np.ndarray) -> float:
        return np.linalg.norm(np.dot(self.A, x) - self.b)

    def df(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.A, x) - self.b
        #return np.dot(self.A.T, np.dot(self.A, x) - self.b)


    def F(self, x: np.ndarray) -> float:
        return self.f(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.df(x)

    def Project(self, x: np.ndarray) -> np.ndarray:
        return self.C.project(x)

    def XToString(self, x: np.ndarray):
        return vectorToString(x)
