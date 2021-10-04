import os
from typing import Union
from typing import Sequence

import numpy as np

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class SLEDirect(VIProblem):
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
        super().__init__(xtest=x_test, x0=x0, C=C, hr_name=hr_name, lam_override=lam_override,
                         lam_override_by_method=lam_override_by_method)

        self.n = p.shape[0]
        self.M = M
        self.p = p

        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = np.zeros(self.n)

    def F(self, x: np.ndarray) -> float:
        return np.linalg.norm(self.M @ x - self.p)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.M.transpose(), np.dot(self.M, x) - self.p)

    def Project(self, x: np.array) -> np.array:
        return self.C.project(x)

    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        np.savetxt(os.path.join(path_to_save, 'M.txt'), self.M)
        np.savetxt(os.path.join(path_to_save, 'p.txt'), self.p)

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest)

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0)