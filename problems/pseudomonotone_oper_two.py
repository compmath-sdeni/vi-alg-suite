import os
from typing import Union
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class PseudoMonotoneOperTwo(VIProblem):
    def __init__(self,
                 *,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, vis: Sequence[VisualParams] = None,
                 hr_name: str = None, unique_name: str = 'PseudoMonotoneOperTwoProblem',
                 x_test: np.ndarray = np.array([0, 0, 0, 0, 0]),
                 lam_override: float = None,
                 lam_override_by_method: dict = None
                 ):
        super().__init__(xtest=x_test, x0=x0, C=C, hr_name=hr_name, unique_name=unique_name,
                         lam_override=lam_override, lam_override_by_method=lam_override_by_method)

        self.arity = 5
        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = np.zeros(self.arity)

        self.M = np.array([
            [5., -1., 2., 0., 2.],
            [-1., 6., -1., 3., 0.],
            [2., -1., 3., 0., 1],
            [0., 3., 0., 5., 0.],
            [2., 0., 1., 0., 4.]
        ])

        self.p = np.array([-1., 2., 1., 0., -1.])

    def F(self, x: np.ndarray) -> float:
        t = self.C.project(x - self.A(x))
        return np.linalg.norm(t - x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        t: float = (np.exp( -(np.linalg.norm(x)**2) ) + 0.1)

        return (np.dot(self.M, x) + self.p)*t

    def Project(self, x: np.array) -> np.array:
        return self.C.project(x)

    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        np.savetxt(os.path.join(path_to_save, 'M.txt'), self.M)
        np.savetxt(os.path.join(path_to_save, 'p.txt'), self.p)

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0)

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest)