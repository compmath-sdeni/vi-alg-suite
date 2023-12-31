import os

import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from methods.projections.simplex_and_ball_proj_inet import euclidean_proj_simplex
from methods.projections.simplex_proj import SimplexProj
from methods.projections.simplex_projection_prom import vec2simplexV2


class ClassicSimplex(ConvexSetConstraints):
    def __init__(self, n: int, b: float = 1.0):
        self.n = n
        self.b = b
        print("ClassicSimplex created. {0}".format(self.toString()))

    def isIn(self, x: np.ndarray) -> bool:
        return abs(x.sum() - self.b) < self.zero_delta

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n - 1):
            res[i] = np.random.rand() * (self.b - res[:i].sum())

        res[self.n - 1] = self.b - res.sum()

        return res

    def project(self, x: np.ndarray) -> np.ndarray:
        return vec2simplexV2(x, self.b)

        # return euclidean_proj_simplex(x, s=self.b)

        # res = x.copy()
        # SimplexProj.doInplace(res, self.b)
        # return res

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([f"n:{self.n}", "\n", f"b:{self.b}"])

    def toString(self):
        return "{0}d classic simplex".format(self.n)
