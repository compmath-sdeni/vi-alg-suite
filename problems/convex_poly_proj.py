import numpy as np

from problems.problem import Problem
from constraints.hyperplanes_bounded_set import HyperplanesBoundedSet
from utils.print_utils import *


class ConvexPolyProjProblem(Problem):
    def __init__(self, *, polytope: HyperplanesBoundedSet, z:np.array,  xtest = None):
        super().__init__(xtest=xtest)
        self.polytope = polytope
        self.z = z

    def x0(self):
        return self.z

    def F(self, x):
        return 0

    def XToString(self, x):
        return vectorToString(x)

    def FValToString(self, v):
        return scalarToString(v)

    def Name(self) -> str:
        pass

    def GetX0(self) -> np.array:
        return self.z

    def GetErrorByTestX(self, xstar) -> float:
        return np.dot((self.xtest - xstar), (self.xtest - xstar)) if self.xtest is not None else xstar
