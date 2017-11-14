from typing import Callable, Sequence
from typing import Union, List

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from problems.splittable_vi_problem import SplittableVIProblem
from problems.visual_params import VisualParams


class LinSysSplitted(SplittableVIProblem):
    def __init__(self, M: int, A: np.ndarray, b: np.ndarray, *, K:int = None,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, vis: Sequence[VisualParams] = None,
                 defaultProjection: np.ndarray = None, xtest: Union[np.ndarray, float] = None, hr_name: str = None, lam_override: float = None):
        super().__init__(M, K=K, xtest=xtest, x0=x0, hr_name=hr_name, C=C, vis=vis, defaultProjection=defaultProjection, lam_override=lam_override)

        self.A: np.ndarray = A
        self.b: np.ndarray = b

        self.f: List[Callable[[np.ndarray], float]] = self.far()
        self.df: List[Callable[[np.ndarray], np.ndarray]] = self.dfar()

    def fi_i(self, i: int) -> Callable[[np.ndarray], float]:
        def f(x:np.ndarray)->float:
            t = (np.dot(self.A[i], x) - self.b[i])**2
            return t

        return f

    def df_i(self, i: int) -> Callable[[np.ndarray], np.ndarray]:
        def df(x: np.ndarray) -> np.ndarray:
            # t = np.zeros(self.M)
            # t[i] = (np.dot(self.A[i], x) - self.b[i])
            t = (np.dot(self.A[i], x) - self.b[i]) * self.A[i]
            return t

        return df

    def far(self) -> List[Callable[[np.ndarray], float]]:
        return [self.fi_i(i) for i in range(self.M)]

    def dfar(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [self.df_i(i) for i in range(self.M)]

    def F(self, x: np.ndarray) -> float:
        return np.linalg.norm(np.dot(self.A, x) - self.b)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A.T @ (np.dot(self.A, x) - self.b)