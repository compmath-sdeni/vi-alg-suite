import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints


# TODO: implement
class HalfSpace(ConvexSetConstraints):
    def __init__(self, n: int):
        self.n: int = n
        print("HalfPlane created. {0}".format(self.toString()))

    def isIn(self, x: np.array) -> bool:
        return False

    def getSomeInteriorPoint(self) -> np.array:
        return np.zeros(self.n)

    def project(self, x: np.array) -> np.array:
        return x

    def toString(self):
        return "n: {0}; bounds: {1}".format(self.n, "???")
