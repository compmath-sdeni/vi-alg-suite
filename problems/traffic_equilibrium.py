import os
from typing import Union, Dict, Optional, Callable

import numpy as np

from constraints.classic_simplex import ClassicSimplex
from methods.projections.simplex_projection_prom import vec2simplexV2
from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class TrafficEquilibrium(VIProblem):
    def __init__(self, *,
                 Gf: Callable[[np.ndarray], np.ndarray],  # Cost function for paths, dependent on traffic
                 d: np.ndarray,  # demands for source-destination pairs
                 W: np.ndarray,  # paths to source-destination pairs incidence matrix
                 x0: Union[np.ndarray] = None,
                 C: ConvexSetConstraints,
                 hr_name: str = None,
                 x_test: np.ndarray = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None
                 ):
        super().__init__(xtest=x_test, x0=x0, C=C, hr_name=hr_name, lam_override=lam_override,
                         lam_override_by_method=lam_override_by_method)

        self.Gf = Gf
        self.d = d
        self.W = W

        self.n: int = W.shape[0]  # paths count

    def F(self, x: np.ndarray) -> float:
        # GAP
        g = np.max(self.Gf(x)) - np.min(self.Gf(x))
        return g  # priciest path expenses vs cheapest path expenses

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        return self.Gf(x)

    def Project(self, x: np.array) -> np.array:
        res = x.copy()
        for i in range(self.d.shape[0]):
            t = res[self.W[i]]
            proj_part = vec2simplexV2(t, self.d[i])
            res[self.W[i]] = proj_part

        return res

    def getIndividualLoss(self, x: np.ndarray) -> float:
        return self.Gf(x).max()

    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        np.savetxt("{0}/{1}".format(path_to_save, 'W.txt'), self.W, delimiter=',', newline="],\n[")
        np.savetxt("{0}/{1}".format(path_to_save, 'd.txt'), self.d, delimiter=',', newline="],\n[")

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest, delimiter=',')

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0, delimiter=',')

        return path_to_save

    def loadFromFile(self, path: str):
        self.W = np.loadtxt("{0}/{1}".format(path, 'W.txt'))
        self.d = np.loadtxt("{0}/{1}".format(path, 'd.txt'))
        self.xtest = np.loadtxt("{0}/{1}".format(path, 'x_test.txt'))

    def GetExtraIndicators(self, x: Union[np.ndarray, float]) -> Optional[Dict]:
        return {
            'Individual loss': self.getIndividualLoss(x),
        }