import math

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints


class HalfSpace(ConvexSetConstraints):
    """Half space bounded by n dimensional hyperplane, with vector equation (a,x) <= b.

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
        return np.dot(self.a, x) <= self.b or math.isclose(np.dot(self.a, x), self.b)

    def getSomeInteriorPoint(self) -> np.array:
        x = np.ones_like(self.a)
        return self.project(x)

    def project(self, x: np.ndarray) -> np.array:
        # if not in half space, project to bounding hyperplane Hn = (c,x)=b
        # Px = x + (b - <c,x>)*c/<c,c>

        if self.isIn(x):
            return x
        else:
            return x + ((self.b - np.dot(self.a, x)) * self.a) / np.dot(self.a, self.a)

    def toString(self):
        return "Halfspace-{0} ({1},x) <= {2}".format(self.a.shape[0], self.a, self.b)
