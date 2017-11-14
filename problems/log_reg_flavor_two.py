from typing import Callable, Sequence
from typing import Union, List
import math

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from problems.splittable_vi_problem import SplittableVIProblem
from problems.visual_params import VisualParams


class LogRegFlavorTwo(SplittableVIProblem):
    def __init__(self, X: np.ndarray, y: np.ndarray, *, K:int = None,
                 w0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, vis: Sequence[VisualParams] = None,
                 defaultProjection: np.ndarray = None, wtest: Union[np.ndarray, float] = None, hr_name: str = None, lam_override: float = None):
        super().__init__(X.shape[0], K=K, xtest=wtest, x0=w0, hr_name=hr_name, C=C, vis=vis, defaultProjection=defaultProjection, lam_override=lam_override)

        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.N = self.X.shape[1]
        self.kM: float = 1.0/self.M

        self.f: List[Callable[[np.ndarray], float]] = self.far()
        self.df: List[Callable[[np.ndarray], np.ndarray]] = self.dfar()

    def fi_i(self, i: int) -> Callable[[np.ndarray], float]:
        def f(w: np.ndarray)->float:
            t = 1.0 - self.y[i]*(w@self.X[i])
            return t if t > 0 else 0

        return f

    def df_i(self, i: int) -> Callable[[np.ndarray], np.ndarray]:
        def df(w: np.ndarray) -> np.ndarray:
            t = 1.0 - self.y[i] * (w @ self.X[i])

            xy = self.y[i]*self.X[i]
            if t > 0:
                return -xy
            elif math.isclose(t, 0):
                res = np.random.rand(self.N) * (-xy)
                return res
            else:
                return np.zeros(self.N)

        return df

    def far(self) -> List[Callable[[np.ndarray], float]]:
        return [self.fi_i(i) for i in range(self.M)]

    def dfar(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [self.df_i(i) for i in range(self.M)]
    #
    # def F(self, x: np.ndarray) -> float:
    #     return np.linalg.norm(np.dot(self.A, x) - self.b, 1)
    #
    # def GradF(self, x: np.ndarray) -> np.ndarray:
    #     # np.sign(A @ x - b) @ A
    #     return np.dot(np.sign(np.dot(self.A, x) - self.b), self.A)
