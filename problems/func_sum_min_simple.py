from typing import Callable, Sequence
from typing import Union, List

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from problems.splittable_vi_problem import SplittableVIProblem
from problems.visual_params import VisualParams


class FuncSumMinSimple(SplittableVIProblem):
    def __init__(self, M: int, f: List[Callable[[np.ndarray], float]], df: List[Callable[[np.ndarray], np.ndarray]], *,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, vis: Sequence[VisualParams] = None,
                 defaultProjection: np.ndarray = None, xtest: Union[np.ndarray, float] = None, hr_name: str = None):
        super().__init__(M, xtest=xtest, x0=x0, C=C, hr_name=hr_name, vis=vis, defaultProjection=defaultProjection)

        self.f: List[Callable[[np.ndarray], float]] = f
        self.df: List[Callable[[np.ndarray], np.ndarray]] = df
