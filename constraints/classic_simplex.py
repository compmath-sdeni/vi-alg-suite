import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from methods.projections.simplex_proj import SimplexProj


class ClassicSimplex(ConvexSetConstraints):
    def __init__(self, n: int):
        self.n = n
        print("ClassicSimplex created. {0}".format(self.toString()))

    def isIn(self, x: np.ndarray) -> bool:
        return x.sum() == 1

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n - 1):
            res[i] = np.random.rand() * (1.0 - res[:i].sum())

        res[n - 1] = 1 - res.sum()

        return res

    def project(self, x: np.ndarray) -> np.ndarray:
        res = x.copy()
        SimplexProj.doInplace(res)
        return res

    def toString(self):
        return "{0}d classic simplex".format(self.n)
