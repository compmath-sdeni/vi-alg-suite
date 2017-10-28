import numpy as np
import itertools
from tools.print_utils import vectorToString
from constraints.convex_set_constraint import ConvexSetConstraints


class Hyperrectangle(ConvexSetConstraints):
    def __init__(self,n: float, bounds: list):
        self.n=n
        if len(bounds) != n:
            if len(bounds) == 1:
                self.bounds = [bounds for x in range(n)]
            else:
                raise Exception("Hyperrectangle - bounds length differs from dimentions! {0} != {1}".format(len(bounds), n))
        else:
            self.bounds = bounds
        print("Hyperrectangle created. {0}".format(self.toString()))

    def _isInSingleDimention(self, x:np.array, dimIndex:int):
        return self.bounds[dimIndex][0] <= x[dimIndex]  <= self.bounds[dimIndex][1]

    def _isInSingleDimention(self, u):
        return (u[1][0] <= u[0] <= u[1][1])

    def isIn(self, x:np.array) -> bool:
        if (x.shape[0] != self.n): raise Exception("Hyperrectangle - isIn - bad vector dimentions: {0} (need {1})".format(x.shape, self.n))

        #return next(itertools.filterfalse(lambda i:b[i][0]<=x[i]<=b[i][1], range(0, self.n)), False) == False
        #return next(itertools.filterfalse(self._isInSingleDimention, zip(x, self.bounds)), False) == False
        #return next(filter(self._isInSingleDimention, zip(x, self.bounds)), False) == False
        return next(filter(lambda u: not (u[1][0] <= u[0] <= u[1][1]), zip(x,self.bounds)), False) == False

    def getSomeInteriorPoint(self) -> np.array:
        return np.array(list(itertools.starmap(lambda a,b: (a+b)*0.5, self.bounds)))

    def project(self,x:np.array) -> np.array:
        if x.shape[0] != self.n: raise Exception(
            "Hyperrectangle - project - bad vector dimentions: {0} (need {1})".format(x.shape, self.n))
        return np.array(list(itertools.starmap(lambda x, bounds : bounds[0] if x<=bounds[0] else (bounds[1] if x>=bounds[1] else x), zip(x, self.bounds))))

    def toString(self):
        bstr = (str(["[{0}, {1}]".format(v[0],v[1]) for v in (self.bounds if len(self.bounds) <= 5 else [self.bounds[0], self.bounds[1], ('..', '..'), self.bounds[-2], self.bounds[-1]])]))
        return "n: {0}; bounds: {1}".format(self.n, bstr)