import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from methods.projections.simplex_proj import SimplexProj


class ClassicSimplex(ConvexSetConstraints):
    def __init__(self, n: int, b: float = 1.0):
        self.n = n
        self.b = b
        print("ClassicSimplex created. {0}".format(self.toString()))

    def isIn(self, x: np.ndarray) -> bool:
        return abs(x.sum() - self.b) < 0.000000001

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n - 1):
            res[i] = np.random.rand() * (self.b - res[:i].sum())

        res[self.n - 1] = self.b - res.sum()

        return res

    def project(self, x: np.ndarray) -> np.ndarray:
        res = x.copy()
        SimplexProj.doInplace(res, self.b)
        return res

    def toString(self):
        return "{0}d classic simplex".format(self.n)
