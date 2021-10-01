import time
from typing import Union

import numpy as np
from scipy import linalg
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod


class Tseng(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, maxLam: float = 0.9, *,
                 min_iters: int = 0, max_iters=5000):
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters)
        self.maxLam = maxLam
        self.x = self.px = self.problem.x0.copy()
        self.D = 0

    def __iter__(self):
        self.x = self.px = self.problem.x0.copy()
        return super().__iter__()

    def doStep(self):
        Ax = self.problem.A(self.x)
        self.operator_count += 1

        y = self.problem.Project(self.x - self.lam * Ax)
        self.projections_count += 1

        self.D = np.linalg.norm(y - self.x)

        if self.D >= self.eps or self.iter < self.min_iters:
            self.px = self.x
            self.x = y - self.lam * (self.problem.A(y) - Ax)
            self.operator_count += 1

    def doPostStep(self):
        self.setHistoryData(x=self.x, step_delta_norm=self.D, goal_func_value=self.problem.F(self.x))

    def isStopConditionMet(self):
        return super(Tseng, self).isStopConditionMet() or self.D < self.eps

    def __next__(self):
        return super(Tseng, self).__next__()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}; MaxLam: {1}".format(self.problem.XToString(self.problem.x0),
                                                                            self.maxLam)

    def currentState(self) -> dict:
        # return dict(super().currentState(), x=([*self.x, self.problem.F(self.x)], [*self.y, self.problem.F(self.y)]), lam=self.lam)
        return dict(super().currentState(), x=(self.x, ), F=(self.problem.F(self.x), ),
                    D=self.D, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))
