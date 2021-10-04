import os

import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from methods.projections.simplex_proj import SimplexProj

# looks erroneus, not a simplex??
class PositiveSimplexArea(ConvexSetConstraints):
    def __init__(self, n: int, b: float = 1.0):
        super().__init__()
        self.n: int = n
        self.b: float = b

    def isIn(self, x: np.ndarray) -> bool:
        return x.sum() <= self.b

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n):
            res[i] = np.random.rand() * (self.b - res[:i].sum())

        return res

    def project(self, x: np.ndarray) -> np.ndarray:
        res: np.ndarray = x.copy()
        for i in range(self.n):
            if res[i] < 0:
                res[i] = 0

        if res.sum() > self.b:
            SimplexProj.doInplace(res, self.b)

        return res

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([f"n\t{self.n}", "\n", f"b\t{self.b}"])

    def toString(self):
        return "{0}d positive simplex area scaled to {1}".format(self.n, self.b)
