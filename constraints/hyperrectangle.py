import math
import os

import numpy as np
import itertools
from utils.print_utils import vectorToString
from constraints.convex_set_constraint import ConvexSetConstraints, ConvexSetConstraintsException


class Hyperrectangle(ConvexSetConstraints):
    def __init__(self, n: float, bounds: list):
        super().__init__()
        self.n = n
        if len(bounds) != n:
            if len(bounds) == 1:
                self.bounds = [bounds for x in range(n)]
            else:
                raise ConvexSetConstraintsException("Hyperrectangle init",
                                                    "Bounds length differs from dimentions! {0} != {1}".
                                                    format(len(bounds), n))
        else:
            self.bounds = bounds

    def _isInSingleDimension(self, x: np.array, dimIndex: int):
        return self.bounds[dimIndex][0] <= x[dimIndex] <= self.bounds[dimIndex][1]

    def getDim(self):
        return self.n

    def isIn(self, x: np.array) -> bool:
        if x.shape[0] != self.n:
            raise ConvexSetConstraintsException("Hyperrectangle isIn",
                                                f"Bad vector dimensions: {x.shape} (need {self.n})")

        #or (math.isclose(u[0], u[1][0]) or math.isclose(u[0], u[1][1]))
        return next(filter(lambda u: not (u[1][0] <= u[0] <= u[1][1]), zip(x, self.bounds)), False) == False

    def getSomeInteriorPoint(self) -> np.array:
        return np.array(list(itertools.starmap(lambda a, b: (a + b) * 0.5, self.bounds)))

    def project(self, x: np.array) -> np.array:
        if x.shape[0] != self.n: raise Exception(
            "Hyperrectangle - project - bad vector dimentions: {0} (need {1})".format(x.shape, self.n))
        return np.array(list(
            itertools.starmap(lambda x, bounds: bounds[0] if x <= bounds[0] else (bounds[1] if x >= bounds[1] else x),
                              zip(x, self.bounds))))

    def saveToDir(self, path: str):
        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([f"n\t{self.n}", "\n"])
            file.writelines([f"{bnd[0]}:{bnd[1]}\n" for bnd in self.bounds])

    def toString(self):
        bstr = (str(["[{0}, {1}]".format(v[0], v[1]) for v in (
            self.bounds if len(self.bounds) <= 5 else [self.bounds[0], self.bounds[1], ('..', '..'), self.bounds[-2],
                                                       self.bounds[-1]])]))
        return f"Hyperrect-{self.n} {bstr}"
