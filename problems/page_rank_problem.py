from constraints.classic_simplex import ClassicSimplex
from constraints.l1_ball import L1Ball
from numpy import inf

from constraints.positive_simplex_surface import PositiveSimplexSurface
from problems.viproblem import VIProblem
import numpy as np
from utils.print_utils import vectorToString


class PageRankProblem(VIProblem):
    def __init__(self, *,
                 GraphMatr: np.ndarray,
                 hr_name: str = None,
                 x0: np.ndarray = None,
                 x_test: np.ndarray = None,
                 ):
        self.M = GraphMatr
        self.m = GraphMatr.shape[0]

        if x0 is None:
            x0 = np.concatenate((np.full(self.m, 1.0 / self.m), np.full(self.m, 1.0 / self.m)))

        super().__init__(xtest=x_test, x0=x0, hr_name=hr_name)

        b = GraphMatr.sum(axis=0)[:, None]
        b[b == 0] = 1

        self.M = (GraphMatr.T / b).T

        self.M1 = self.M.T - np.eye(self.m)
        self.M2 = self.M - np.eye(self.m)

        self.simplex = ClassicSimplex(self.m)
        self.ball = L1Ball(self.m, 1)

    @classmethod
    def CreateRandom(cls, m: int, zeroPercent: float = 0) :
        """

        :param m: Nodes count
        :param zeroPercent: Probability of zero element
        """
        res = PageRankProblem(np.ones((m,m), float))

        B = np.random.rand(m, m)
        np.fill_diagonal(B, 0)
        B[B < zeroPercent] = 0
        M = np.ones_like(B)
        M *= 1.0/m

        b = B.sum(axis=0)[:, None]
        b[b == 0] = 1

        res.M = (B.T / b).T

        return res

    def save(self, basePath: str = 'data'):
        np.savetxt("{0}/{1}_M.txt".format(basePath, self.__class__.__name__), self.M, fmt='%.3f')

    def load(self, basePath: str = 'data'):
        self.M = np.loadtxt("{0}/{1}_M.txt".format(basePath, self.__class__.__name__))
        self.m = self.M.shape[0]

    def xpart(self, xy: np.ndarray):
        return xy[:self.m]

    def ypart(self, xy: np.ndarray):
        return xy[self.m:]

    def A(self, x: np.ndarray) -> np.ndarray:
        p1 = self.M1 @ x[self.m:]
        p2 = self.M2 @ x[:self.m]
        return np.concatenate((p1, -p2))
        #return np.concatenate((self.M.T @ self.ypart(x) - self.ypart(x), -(self.M @ self.xpart(x) - self.xpart(x))))
        #return (self.M - np.eye(self.M.shape[0], self.M.shape[0])) @ x

    def F(self, x: np.ndarray) -> float:
        return np.linalg.norm(np.dot(self.M, self.xpart(x)) - self.xpart(x), inf)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def bregmanProject(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        res = np.empty_like(x, dtype=float)

        t = x[:self.m] * np.exp(a[:self.m])
        res[:self.m] = t / t.sum(0, keepdims=True)

        res[self.m:] = self.ball.project(x[self.m:] + a[self.m:])

        return res

    def Project(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate((self.simplex.project(x[:self.m]), self.ball.project(x[self.m:])))

    def XToString(self, x: np.ndarray):
        return vectorToString(x)
