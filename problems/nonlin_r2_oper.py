from problems.viproblem import VIProblem
import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from constraints.allspace import Rn
from utils.print_utils import vectorToString


# noinspection PyPep8Naming
class NonlinR2Oper(VIProblem):
    def __init__(self, q: np.ndarray = None, C: ConvexSetConstraints = None,  x0: np.ndarray = None, hr_name: str= None):
        super().__init__(x0=x0 if x0 is not None else np.array([1,1]), hr_name=hr_name, )
        self.q: np.ndarray = q if q is not None else (np.random.rand(2)*10) - 5
        self.C = C

    def f(self, x: np.ndarray) -> float:
        return 0

    def df(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0]+x[1]+np.sin(x[0]) + self.q[0], -x[0]+x[1]+np.sin(x[1])+self.q[1]])

    def F(self, x: np.ndarray) -> float:
        return self.f(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.df(x)

    def Project(self, x: np.ndarray) -> np.ndarray:
        return self.C.project(x) if self.C is not None else x

    def XToString(self, x: np.ndarray):
        return vectorToString(x)
