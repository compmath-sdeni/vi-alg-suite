from typing import Union, Optional, Dict

from constraints.classic_simplex import ClassicSimplex
from constraints.l1_ball import L1Ball
from numpy import inf

from constraints.positive_simplex_surface import PositiveSimplexSurface
from problems.viproblem import VIProblem
import numpy as np
from utils.print_utils import vectorToString


class PageRankProblem2Simplex(VIProblem):
    def __init__(self, *,
                 GraphMatr: np.ndarray,
                 node_labels: list = None,
                 hr_name: str = None,
                 x0: np.ndarray = None,
                 x_test: np.ndarray = None,
                 lam_override_by_method: dict = None
                 ):
        self.simplex1 = None
        self.simplex2 = None
        self.M = None
        self.m = None

        m = GraphMatr.shape[0]

        self.node_labels = node_labels

        if x0 is None:
            x0 = np.concatenate((np.full(m, 1.0 / m), np.full(m, 1.0 / m)))

        super().__init__(xtest=x_test, x0=x0, hr_name=hr_name, lam_override_by_method=lam_override_by_method)

        b = GraphMatr.sum(axis=0)[:, None]
        b[b == 0] = 1
        M = (GraphMatr.T / b).T
        self.init_from_Markov_matrix(M)

    def init_from_Markov_matrix(self, M: np.ndarray):
        self.m = M.shape[0]
        self.M = M
        I = np.eye(self.m)
        # J should be a matrix n x 2n in form (I, -I)
        J = np.concatenate((I, -I), axis=1)

        A1 = J.T @ (I - self.M)
        A2 = (self.M.T - I) @ J

        # A should be a matrix 3n x 3n in form (A1, 0 / 0, A2)
        self.AComb = np.zeros((self.m * 3, self.m * 3))
        self.AComb[:self.m*2, :self.m] = A1
        self.AComb[self.m*2:, self.m:] = A2

        self.simplex1 = ClassicSimplex(self.m)
        self.simplex2 = ClassicSimplex(self.m*2)

    def save(self, basePath: str = 'data'):
        np.savetxt("{0}/{1}_M.txt".format(basePath, self.__class__.__name__), self.M, fmt='%.3f')

    def load(self, basePath: str = 'data'):
        self.M = np.loadtxt("{0}/{1}_M.txt".format(basePath, self.__class__.__name__))
        self.init_from_Markov_matrix(self.M)

    def xpart(self, xy: np.ndarray):
        return xy[:self.m]

    def ypart(self, xy: np.ndarray):
        return xy[self.m:]

    def A(self, x: np.ndarray) -> np.ndarray:
        p = self.AComb @ x
        return p

    def F(self, x: np.ndarray) -> float:
        return np.linalg.norm(np.dot(self.M, self.xpart(x)) - self.xpart(x), inf)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def bregmanProject(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:

        res = np.empty_like(x, dtype=float)

        t = x[:self.m] * np.exp(a[:self.m])
        res[:self.m] = t / t.sum(0, keepdims=True)

        t = x[self.m:] * np.exp(a[self.m:])
        res[self.m:] = t / t.sum(0, keepdims=True)

        return res

    def Project(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate((self.simplex1.project(x[:self.m]), self.simplex2.project(x[self.m:])))

    def XToString(self, x: np.ndarray):
        return vectorToString(x)

    def GetExtraIndicators(self, x: Union[np.ndarray, float], *, averaged_x: np.ndarray = None, final: bool = False) -> \
    Optional[Dict]:
        res = {
            'x elems sum': np.sum(x[:self.m]),
            'y elems sum': np.sum(x[self.m:])
        }

        if final:
            top_n: int = 10
            if self.node_labels is not None:
                res['top ranks'] = self.node_labels[np.argsort(x[:self.m])[::-1][:top_n]]
            else:
                res['top ranks'] = np.argsort(x[:self.m])[::-1][:top_n]

            if averaged_x is not None:
                if self.node_labels is not None:
                    res['top ranks from AVG'] = self.node_labels[np.argsort(averaged_x[:self.m])[::-1][:top_n]]
                else:
                    res['top ranks from AVG'] = np.argsort(averaged_x[:self.m])[::-1][:top_n]

        return res
