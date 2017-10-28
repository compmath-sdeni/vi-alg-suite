import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints


class Rn(ConvexSetConstraints):
    def __init__(self, n: int):
        self.n = n

    def isIn(self, x: np.ndarray) -> bool:
        return True

    def getSomeInteriorPoint(self) -> np.ndarray:
        v = np.random.ranf(self.n)
        return v/v.sum()

    def project(self, x: np.ndarray) -> np.ndarray:
        return x

    def toString(self):
        return "n: {0}".format(self.n)
