import os

import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints


class RnPlus(ConvexSetConstraints):
    def __init__(self, n: int):
        self.n = n

    def isIn(self, x: np.ndarray) -> bool:
        return not (np.any(x < 0))

    def getSomeInteriorPoint(self) -> np.ndarray:
        v = np.random.rand(self.n) * 100.
        return v

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0, x)

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.write(f"n:{self.n}")

    def toString(self):
        return "R^n_+, n: {0}".format(self.n)
