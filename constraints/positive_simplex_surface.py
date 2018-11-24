import numpy as np
from constraints.positive_simplex_area import PositiveSimplexArea
from methods.projections.simplex_proj import SimplexProj
from methods.projections.simplex_projection_prom import euclidean_proj_simplex


class PositiveSimplexSurface(PositiveSimplexArea):
    def __init__(self, n: int, b: float = 1.0, delta: float = 0.000000001):
        super().__init__(n, b)
        self.delta = delta

    def isIn(self, x: np.ndarray) -> bool:
        return (not (x<0).any()) and (abs(x.sum() - self.b) < self.delta)

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n - 1):
            res[i] = np.random.rand() * (self.b - res[:i].sum())

        res[self.n - 1] = (self.b - res[:self.n - 1].sum())

        return res

    def project(self, x: np.ndarray) -> np.ndarray:

        # using prom algho
        return euclidean_proj_simplex(x, s=self.b)

        # using self-coded algho
        # res: np.ndarray = x.copy()
        #
        # if not self.isIn(res):
        #     SimplexProj.doInplace(res, self.b)
        #
        # return res

    def toString(self):
        return "{0}d positive simplex surface scaled to {1}".format(self.n, self.b)
