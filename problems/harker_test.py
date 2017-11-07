from problems.viproblem import VIProblem
import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from constraints.allspace import Rn
from utils.print_utils import vectorToString


# noinspection PyPep8Naming
class HarkerTest(VIProblem):
    def __init__(self, M: int, C: ConvexSetConstraints = None,
                 x0: np.ndarray = None, hr_name: str = None):
        super().__init__(x0=x0 if x0 is not None else np.ones(M), hr_name=hr_name)

        B = np.random.rand(M, M) * 5 - 10
        S = np.random.rand(M, M) * 5 - 10
        for i in range(M):
            S[i, i] = 0;

        for i in range(M):
            for j in range(M):
                S[i, j] = -S[j, i];

        D = np.identity(M, float)
        for i in range(M):
            D[i, i] = np.random.randint(0, 1) * 0.3

        self.A = B @ B.T + S + D
        self.C = C

        # self.q = np.random.rand(M) * -500.0
        self.q = np.zeros(M)

        self.norm = np.linalg.norm(self.A, 2)
        print("Norm: ", self.norm)

        # r = np.linalg.eig(self.A)
        # print("A:\n", self.A)
        # print("Eig:\n", r)

    def f(self, x: np.ndarray) -> float:
        return np.dot(x, x)

    def df(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.A, x) + self.q

    def F(self, x: np.ndarray) -> float:
        return self.f(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.df(x)

    def Project(self, x: np.ndarray) -> np.ndarray:
        return self.C.project(x) if self.C is not None else x

    def XToString(self, x: np.ndarray):
        return vectorToString(x)
