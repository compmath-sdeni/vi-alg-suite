from typing import Union
from typing import Sequence

import numpy as np

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class SLESaddle(VIProblem):
    def __init__(self, *,
                 M: np.ndarray, p: np.ndarray,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints,
                 vis: Sequence[VisualParams] = None,
                 hr_name: str = None,
                 x_test: np.ndarray = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None
                 ):
        super().__init__(xtest=x_test, x0=x0, hr_name=hr_name, lam_override=lam_override,
                         lam_override_by_method=lam_override_by_method)

        self.n = M.shape[0]
        self.m = M.shape[1]

        self.C = C

        self.M = M
        self.p = p

        if self._x0.shape[0] == self.m:
            self._x0 = np.concatenate((self._x0, np.ones(self.n)))

        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = np.zeros(self.m)

    def F(self, x: np.ndarray) -> float:
        return np.linalg.norm(self.M @ x[:self.m] - self.p)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        p1 = np.dot(self.M.transpose(), x[self.m:])
        p2 = self.p - np.dot(self.M, x[:self.m])
        return np.concatenate((p1, p2))

    def Project(self, x: np.array) -> np.array:
        p1 = self.C.project(x[:self.m])
        p2 = x[self.m:]
        nr = np.linalg.norm(p2)
        if nr > 1.:
            p2 /= nr

        return np.concatenate((p1, p2))
