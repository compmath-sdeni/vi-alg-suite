import time
from typing import Union

import numpy as np
from scipy import linalg
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod


class KorpelevichMod(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, maxLam: float = 0.9, *,
                 min_iters: int = 0, max_iters=5000):
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters)
        self.maxLam = maxLam
        self.x = self.px = self.y = self.problem.x0.copy()
        self.D = 0

    def __iter__(self):
        self.x = self.px = self.problem.x0.copy()
        return super().__iter__()

    def doStep(self):
        Ax = self.problem.A(self.x)
        self.operator_count += 1

        self.y: Union[np.ndarray, float] = self.problem.Project(self.x - self.lam * Ax)
        self.projections_count += 1

        Ay = self.problem.A(self.y)
        self.operator_count += 1

        if self.iter > 0 and (linalg.norm(self.x - self.y) >= self.zero_delta):
            t = 0.9 * linalg.norm(self.x - self.y) / linalg.norm(
                Ax - Ay)
            self.lam = t if t <= self.maxLam else self.maxLam
        else:
            self.lam = self.maxLam

        self.px, self.x = self.x, self.problem.Project(self.x - self.lam * Ay)
        self.projections_count += 1

    def doPostStep(self):
        self.setHistoryData(x=self.x, y=self.y, step_delta_norm=self.D, goal_func_value=self.problem.F(self.x))

    def isStopConditionMet(self):
        return super(KorpelevichMod, self).isStopConditionMet() or self.D < self.eps

    def __next__(self):
        self.D = linalg.norm(self.x - self.y)
        return super(KorpelevichMod, self).__next__()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}; MaxLam: {1}".format(self.problem.XToString(self.problem.x0),
                                                                            self.maxLam)

    def currentState(self) -> dict:
        # return dict(super().currentState(), x=([*self.x, self.problem.F(self.x)], [*self.y, self.problem.F(self.y)]), lam=self.lam)
        return dict(super().currentState(), x=(self.x, self.y), F=(self.problem.F(self.x), self.problem.F(self.y)),
                    D=self.D, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))
