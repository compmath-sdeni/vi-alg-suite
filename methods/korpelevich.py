from typing import Union
import time

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod
import numpy as np
from scipy import linalg


class Korpelevich(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, min_iters = 0, max_iters = 5000):
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters)
        self.px: Union[np.ndarray, float] = self.x.copy()
        self.D: float = 0
        self.y: Union[np.ndarray, float] = self.x.copy()

    def __iter__(self):
        self.px = self.x.copy()
        self.D = 0
        return super().__iter__()

    def __next__(self):
        return super(Korpelevich, self).__next__()
    
    def doStep(self):
        y: np.ndarray = self.problem.Project(self.x - self.lam * self.problem.A(self.x))
        self.projections_count += 1
        self.operator_count += 1

        self.px = self.x
        self.x = self.problem.Project(self.x - self.lam * self.problem.A(y))
        self.projections_count += 1
        self.operator_count += 1

        self.D = linalg.norm(self.x - y)

    def doPostStep(self):
        self.setHistoryData(x=self.x, y=self.y, step_delta_norm=self.D, goal_func_value=self.problem.F(self.x))

    def isStopConditionMet(self):
        return super(Korpelevich, self).isStopConditionMet() or self.D < self.eps

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, self.y), D=(self.D, linalg.norm(self.x - self.px)),
                    F=(self.problem.F(self.x), self.problem.F(self.y)), iterEndTime = self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: D: {1}; x: {2}; y: {3}; F: {4}".format(
            self.iter, self.D, self.problem.XToString(self.x),
            self.problem.XToString(self.y), self.problem.FValToString(self.problem.F(self.x)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString()+"; x0: {0}".format(self.problem.XToString(self.problem.x0))
