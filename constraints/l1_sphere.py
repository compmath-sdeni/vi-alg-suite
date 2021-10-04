import os

import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from methods.projections.simplex_projection_prom import euclidean_proj_l1ball


class L1Sphere(ConvexSetConstraints):
    def __init__(self, n: int, b: float = 1.0):
        super().__init__()
        self.n: int = n
        self.b: float = b

    def isIn(self, x: np.ndarray) -> bool:
        return np.fabs(np.sum(np.abs(x)) - self.b) < self.zero_delta

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        r: float = 0
        for i in range(self.n - 1):
            r = (self.b - abs(res[:i]).sum())
            if r > self.zero_delta:
                res[i] = ((np.random.rand() * 2.0 - 1) * self.b) * r
            else:
                res[i] = 0

        if r > self.zero_delta:
            res[self.n - 1] = (self.b - abs(res[:self.n - 1]).sum()) * (1 if np.random.rand() > 0.5 else -1)

        return res

    def project(self, x: np.ndarray) -> np.ndarray:

        # using prom algo
        return euclidean_proj_l1ball(x, s=self.b)

        # using self-coded algo
        # res: np.ndarray = x.copy()
        #
        # if not self.isIn(res):
        #     neg: np.ndarray = res < 0
        #     res[neg] *= -1
        #     SimplexProj.doInplace(res, self.b)
        #     res[neg] *= -1
        #
        # return res

    def getDim(self):
        return self.n

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([f"n\t{self.n}", "\n", f"b\t{self.b}"])

    def toString(self):
        return "{0}d L1 sphere R={1}".format(self.n, self.b)
