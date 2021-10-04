import math
import os

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints


class Hyperplane(ConvexSetConstraints):
    """n dimensional hyperplane, with vector equation (a,x)=b.

        Attributes:
            a -- normal vector
            b -- right side
        """

    def __init__(self, *, a: np.ndarray, b: float):
        """
        :param a: normal vector
        :param b: right side of (a,x)=b
        """
        super().__init__()

        self.a = a
        self.b = b

    def getDim(self):
        return self.a.shape[0]

    def isIn(self, x: np.array) -> bool:
        if self.b != 0:
            return math.isclose(np.dot(self.a, x), self.b)
        else:
            return math.fabs(np.dot(self.a, x)) < self.zero_delta

    def getSomeInteriorPoint(self) -> np.array:
        x = np.ones_like(self.a)
        return self.project(x)

    def project(self, x: np.ndarray) -> np.array:
        # project to Hn = (c,x)<=b
        # Px = x + (b - <c,x>)*c/<c,c>

        if self.isIn(x):
            return x
        else:
            return x + ((self.b - np.dot(self.a, x)) * self.a) / np.dot(self.a, self.a)

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([str(self.b), "\n", np.array2string(self.a, max_line_width=100000)])

    def toString(self):
        return "Hyperplane-{0} ({1},x)={2}".format(self.a.shape[0], self.a, self.b)
