import os

import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints


class L2Ball(ConvexSetConstraints):
    def __init__(self, n: int, r: float = 1.0):
        super().__init__()
        self.n: int = n
        self.r: float = r

    def isIn(self, x: np.ndarray) -> bool:
        return np.linalg.norm(x) <= self.r

    def getSomeInteriorPoint(self) -> np.ndarray:
        res: np.ndarray = np.random.rand(self.n)*self.r

        if self.isIn(res):
            return res
        else:
            k = np.linalg.norm(res)
            return res/k

    def getDim(self):
        return self.n

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.isIn(x):
            return x
        else:
            k = np.linalg.norm(x)
            return x/k

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([f"n\t{self.n}\n", "\n", f"r\t{self.r}"])

    def toString(self):
        return "{0}d L2 ball R={1}".format(self.n, self.r)
