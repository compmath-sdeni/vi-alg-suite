import os
from typing import Union
from typing import Sequence

import numpy as np

from constraints.classic_simplex import ClassicSimplex
from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class MinMaxGame(VIProblem):
    def __init__(self, *,
                 P: np.ndarray,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints,
                 vis: Sequence[VisualParams] = None,
                 hr_name: str = None,
                 x_test: np.ndarray = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None
                 ):
        super().__init__(xtest=x_test, x0=x0, C=C, hr_name=hr_name, lam_override=lam_override,
                         lam_override_by_method=lam_override_by_method)

        self.m = P.shape[1]
        self.n = P.shape[0]

        self.C1 = ClassicSimplex(self.m)
        self.C2 = ClassicSimplex(self.n)

        self.P = P

        if self._x0.shape[0] == self.m:
            self._x0 = np.concatenate((self._x0, np.ones(self.n, dtype=float)))

        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = np.zeros(self.m + self.n)

    def F(self, x: np.ndarray) -> float:
        # GAP
        return np.max(self.P @ x[:self.m]) - np.min(self.P.T @ x[self.m:])
        # ???
        # return np.dot(self.P.T @ x[self.m:], x[:self.m]) - np.dot(self.P @ x[:self.m], x[self.m:])

        # Game value
        # return np.dot(self.P @ x[:self.m], x[self.m:])

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        p1 = self.P.T @ x[self.m:]
        p2 = -self.P @ x[:self.m]
        return np.concatenate((p1, p2))

    def Project(self, x: np.array) -> np.array:
        p1 = self.C1.project(x[:self.m])
        p2 = self.C2.project(x[self.m:])

        return np.concatenate((p1, p2))

    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        np.savetxt("{0}/{1}".format(path_to_save, 'P.txt'), self.P, delimiter=',', newline="],\n[")

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest, delimiter=',')

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0, delimiter=',')

        return path_to_save

    def loadFromFile(self, path: str):
        self.P = np.loadtxt("{0}/{1}".format(path, 'P.txt'))
        self.xtest = np.loadtxt("{0}/{1}".format(path, 'x_test.txt'))
