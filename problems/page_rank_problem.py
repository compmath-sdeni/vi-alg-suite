from problems.viproblem import VIProblem
import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from constraints.allspace import Rn
from tools.print_utils import vectorToString


class PageRankProblem(VIProblem):
    def __init__(self, GraphMatr: np.ndarray):
        self.A = GraphMatr
        m = GraphMatr.shape[0]
        super().__init__(x0=np.full(m, 1.0 / m))

        b = GraphMatr.sum(axis=0)[:, None]
        b[b == 0] = 1

        self.A = (GraphMatr.T / b).T

    @classmethod
    def CreateRandom(cls, m: int, zeroPercent: float = 0) :
        """

        :param m: Nodes count
        :param zeroPercent: Probability of zero element
        """
        res = PageRankProblem(np.ones((m,m), float))

        A = np.random.rand(m, m)
        np.fill_diagonal(A, 0)
        A[A < zeroPercent] = 0
        M = np.ones_like(A)
        M *= 1.0/m

        b = A.sum(axis=0)[:, None]
        b[b == 0] = 1

        res.A = (A.T / b).T

        return res

    def save(self, basePath: str = 'data'):
        np.savetxt("{0}/{1}_A.txt".format(basePath, self.__class__.__name__), self.A, fmt='%.3f')

    def load(self, basePath: str = 'data'):
        self.A = np.loadtxt("{0}/{1}_A.txt".format(basePath, self.__class__.__name__))

    def f(self, x: np.ndarray) -> float:
        return np.linalg.norm(np.dot(self.A, x) - x)

    def df(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.A, x) - x

    def F(self, x: np.ndarray) -> float:
        return self.f(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.df(x)

    def Project(self, x: np.ndarray) -> np.ndarray:
        return x

    def XToString(self, x: np.ndarray):
        return vectorToString(x)
