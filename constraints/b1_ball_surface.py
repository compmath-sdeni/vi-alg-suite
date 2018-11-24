import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from methods.projections.simplex_proj import SimplexProj


class B1BallSurface(ConvexSetConstraints):
    def __init__(self, n: int, b: float = 1.0, delta: float = 0.000000001):
        super().__init__()
        self.n: int = n
        self.b: float = b
        self.delta = delta

    def isIn(self, x: np.ndarray) -> bool:
        return abs(abs(x).sum() - self.b) < self.delta

    def getSomeInteriorPoint(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n - 1):
            r = (self.b - abs(res[:i]).sum())
            if r > self.delta:
                res[i] = ((np.random.rand()*2.0 - 1)*self.b) * r
            else:
                res[i] = 0

        if r > self.delta:
            res[self.n - 1] = (self.b - abs(res[:self.n - 1]).sum()) * (1 if np.random.rand()>0.5 else -1)

        return res

    def project(self, x: np.ndarray) -> np.ndarray:
        res: np.ndarray = x.copy()

        if not self.isIn(res):
            neg: np.ndarray = res < 0
            res[neg] *= -1
            SimplexProj.doInplace(res, self.b)
            res[neg] *= -1

        return res

    def toString(self):
        return "{0}d B1 ball surface scaled to {1}".format(self.n, self.b)
