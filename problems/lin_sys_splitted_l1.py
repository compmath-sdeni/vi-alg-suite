from typing import Callable, Sequence
from typing import Union, List
import math

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from problems.splittable_vi_problem import SplittableVIProblem
from problems.visual_params import VisualParams


class LinSysSplittedL1(SplittableVIProblem):
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
            t = np.abs(np.dot(self.A[i], x) - self.b[i])
            return t

        return f

    def df_i(self, i: int) -> Callable[[np.ndarray], np.ndarray]:
        def df(x: np.ndarray) -> np.ndarray:
            t = np.dot(self.A[i],  x) - self.b[i]
            return np.sign(t) * self.A[i] if not math.isclose(t, 0) else (np.random.ranf()*2.0 - 1)*self.A[i]

        return df

    def far(self) -> List[Callable[[np.ndarray], float]]:
        return [self.fi_i(i) for i in range(self.M)]

    def dfar(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [self.df_i(i) for i in range(self.M)]

    def F(self, x: np.ndarray) -> float:
        return np.linalg.norm(np.dot(self.A, x) - self.b, 1)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        # np.sign(A @ x - b) @ A
        return np.dot(np.sign(np.dot(self.A, x) - self.b), self.A)
