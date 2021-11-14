import time
from typing import Union

import numpy as np
from scipy import linalg
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod


class MalitskyTam(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, x1: np.ndarray,
                 min_iters: int = 0, max_iters=5000, hr_name: str = None):
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters, hr_name=hr_name)

        self.ppx = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.x = self.x1 = x1

        self.Apx = self.problem.A(self.px)
        self.Ax = self.problem.A(self.x)

        self.D:float = 0
        self.D2:float = 0

    def __iter__(self):
        self.ppx = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.x = self.x1.copy()

        self.D = 0
        self.D2 = 0

        self.Apx = self.problem.A(self.px)
        self.Ax = self.problem.A(self.x)
        self.operator_count += 2

        return super().__iter__()

    def doStep(self):
        self.ppx = self.px
        self.px = self.x
        self.x = self.problem.Project(self.x - self.lam * self.Ax - self.lam * (self.Ax - self.Apx))
        self.projections_count += 1

        self.Apx = self.Ax
        self.Ax = self.problem.A(self.x)
        self.operator_count += 1

        self.D = np.linalg.norm(self.x - self.px)
        self.D2 = np.linalg.norm(self.px - self.ppx)

    def doPostStep(self):
        self.setHistoryData(x=self.x, step_delta_norm=self.D + self.D2, goal_func_value=self.problem.F(self.x))

    def isStopConditionMet(self):
        return super(MalitskyTam, self).isStopConditionMet() or (self.D + self.D2 < self.eps)

    def __next__(self):
        return super(MalitskyTam, self).__next__()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}".format(self.problem.XToString(self.problem.x0))

    def currentState(self) -> dict:
        # return dict(super().currentState(), x=([*self.x, self.problem.F(self.x)], [*self.y, self.problem.F(self.y)]), lam=self.lam)
        return dict(super().currentState(), x=(self.x), F=(self.problem.F(self.x)),
                    D=self.D, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))
