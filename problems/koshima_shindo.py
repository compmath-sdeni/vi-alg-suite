import numpy as np

from methods.projections.simplex_proj import SimplexProj
from problems.viproblem import VIProblem
from utils.print_utils import vectorToString


# noinspection PyPep8Naming
class KoshimaShindo(VIProblem):
    def __init__(self, x0: np.ndarray = None, hr_name: str= None):
        super().__init__(x0=x0 if x0 is not None else np.ones(4), hr_name=hr_name)

    def f(self, x: np.ndarray) -> float:
        return np.dot(x,x)

    def df(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            3.0 * x[0]**2 + 2.0*x[0]*x[1] + 2.0 * x[1]**2 + x[2] + 3.0*x[3]-6.0,
            2.0 * x[0]**2 + x[1] + x[2]**2 + 10.0*x[2] + 2.0 * x[3] - 2.0,
            3.0 * x[0] ** 2 + x[0] * x[1] + 2.0 * x[1]**2 + 2.0*x[2] + 9.0 * x[3] - 9.0,
            x[0]**2 + 3.0 * x[1]**2 + 2.0 * x[2] + 3.0 * x[3] - 3.0
        ], dtype=float)

    def F(self, x: np.ndarray) -> float:
        return self.f(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.df(x)

    def Project(self, x: np.ndarray) -> np.ndarray:
        t = x.copy()
        SimplexProj.doInplace(t, 4.0)
        return t

    def XToString(self, x: np.ndarray):
        return vectorToString(x)
