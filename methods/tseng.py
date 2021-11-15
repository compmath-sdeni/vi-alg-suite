import time
from typing import Union

import numpy as np
from scipy import linalg
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class Tseng(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters: int = 0, max_iters=5000,
                 hr_name: str = None, projection_type: ProjectionType = ProjectionType.EUCLID):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type)

        self.x = self.px = self.problem.x0.copy()
        self.y = np.zeros_like(self.x)
        self.D: float = 0
        self.cum_y = np.zeros_like(self.x)

    def __iter__(self):
        self.x = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.D: float = 0
        return super().__iter__()

    def doStep(self):
        Ax = self.problem.A(self.x)
        self.operator_count += 1

        if self.projection_type == ProjectionType.BREGMAN:
            self.y = self.problem.bregmanProject(self.x, -self.lam * Ax)
        else:
            self.y = self.problem.Project(self.x - self.lam * Ax)

        self.projections_count += 1

        self.cum_y += self.y

        if self.projection_type == ProjectionType.BREGMAN:
            self.D = np.linalg.norm(self.y - self.x, 1)
        else:
            self.D = np.linalg.norm(self.y - self.x)

        if self.D >= self.eps or self.iter < self.min_iters:
            self.px = self.x

            if self.projection_type == ProjectionType.BREGMAN:
                self.x = np.exp(((np.log(self.y) + 1) - self.lam * (self.problem.A(self.y) - Ax))-1)
            else:
                self.x = self.y - self.lam * (self.problem.A(self.y) - Ax)

            self.operator_count += 1

    def doPostStep(self):
        val_for_gap = self.cum_y/(self.iter+1)
        # t = self.problem.F(val_for_gap)
        self.setHistoryData(x=self.x, y=val_for_gap, step_delta_norm=self.D,
                            goal_func_value=self.problem.F(self.y), goal_func_from_average=self.problem.F(val_for_gap))

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
