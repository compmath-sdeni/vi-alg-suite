from typing import Callable, Sequence
from typing import Union, List
import math

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from problems.splittable_vi_problem import SplittableVIProblem
from problems.visual_params import VisualParams


class LogRegFlavorOne(SplittableVIProblem):
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
            t = self.kM * np.log(1.0 + np.exp(-self.y[i]*(w@self.X[i])))
            return t

        return f

    def df_i(self, i: int) -> Callable[[np.ndarray], np.ndarray]:
        def df(w: np.ndarray) -> np.ndarray:
            t = self.kM * self.y[i] * self.X[i] * (1.0/(1.0 + np.exp(-self.y[i] * (w @ self.X[i]))) - 1.0)
            return t

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
