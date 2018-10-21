from typing import Union
import time

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod
import numpy as np
from scipy import linalg


class GradProj(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, min_iters: int = 0):
        super().__init__(problem, eps, lam, min_iters=min_iters)
        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.px: Union[np.ndarray, float] = self.x.copy()
        self.D: float = None

    def __iter__(self):
        self.x = self.problem.x0.copy()
        self.px = self.x.copy()
        self.D = None

        return super().__iter__()

    def __next__(self) -> dict:
        if self.min_iters > self.iter or self.iter == 0 or self.D >= self.eps:
            self.iter += 1
            self.px, self.x = self.x, self.problem.Project(self.x - self.lam * self.problem.GradF(self.x))
            self.D = linalg.norm(self.x - self.px)

            self.iterEndTime = time.process_time()

            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x,), D=self.D, F=(self.problem.F(self.x), ),
                    iterEndTime = self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: D: {1}; x: {2}; F(x): {3}".format(
            self.iter, self.D, self.problem.XToString(self.x), self.problem.FValToString(self.problem.F(self.x)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}".format(self.problem.XToString(self.problem.x0))
