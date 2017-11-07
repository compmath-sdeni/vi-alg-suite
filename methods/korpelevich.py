from typing import Union

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod
import numpy as np
from scipy import linalg


class Korpelevich(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, min_iters = 0):
        super().__init__(problem, eps, lam, min_iters=min_iters)
        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.px: Union[np.ndarray, float] = self.x.copy()
        self.D: float = 0
        self.y: Union[np.ndarray, float] = self.x.copy()

    def __iter__(self):
        self.x = self.problem.x0.copy()
        self.px = self.x.copy()
        self.D: float = 0
        return super().__iter__()

    def __next__(self):
        #self.D = linalg.norm(self.x - self.px)
        self.D = linalg.norm(self.x - self.y)
        if self.min_iters > self.iter or self.D >= self.eps or self.iter == 0:
            self.iter += 1
            self.y: Union[np.ndarray, float]  = self.problem.Project(self.x - self.lam * self.problem.GradF(self.x))
            self.px, self.x = self.x, self.problem.Project(self.x - self.lam * self.problem.GradF(self.y))
            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, self.y), D=self.D,
                    F=(self.problem.F(self.x), self.problem.F(self.y)))

    def currentStateString(self) -> str:
        return "{0}: D: {1}; x: {2}; y: {3}; F: {4}".format(
            self.iter, self.D, self.problem.XToString(self.x),
            self.problem.XToString(self.y), self.problem.FValToString(self.problem.F(self.x)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString()+"; x0: {0}".format(self.problem.XToString(self.problem.x0))
