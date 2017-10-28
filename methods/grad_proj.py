from typing import Union
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod
import numpy as np
from scipy import linalg


class GradProj(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1):
        super().__init__(problem, eps, lam)
        self.x: Union[np.ndarray, float] = self.problem.x0
        self.px: Union[np.ndarray, float] = self.x
        self.D: float = 0

    def __iter__(self):
        self.x = self.problem.x0
        self.px = self.x

        return super().__iter__()

    def __next__(self) -> dict:
        self.D = linalg.norm(self.x - self.px)
        if self.min_iters > self.iter or self.iter == 0 or self.D >= self.eps:
            self.iter += 1
            self.px, self.x = self.x, self.problem.Project(self.x - self.lam * self.problem.GradF(self.x))
            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x,), D=self.D, F=self.problem.F(self.x))

    def currentStateString(self) -> str:
        return "{0}: x: {1}; F(x): {2}".format(
            self.iter, self.problem.XToString(self.x), self.problem.FValToString(self.problem.F(self.x)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}".format(self.problem.XToString(self.problem.x0))
