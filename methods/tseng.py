import time
from typing import Union

import numpy as np
from scipy import linalg

from methods.algorithm_params import StopCondition
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class Tseng(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters: int = 0, max_iters=5000,
                 hr_name: str = None, projection_type: ProjectionType = ProjectionType.EUCLID,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type, stop_condition=stop_condition)

        self.x = self.px = self.problem.x0.copy()

        # to correctly show gap etc on iteration - 0
        self.y = self.problem.x0.copy() # np.zeros_like(self.x)

        self.D: float = 0
        self.cum_y = np.zeros_like(self.x)
        self.averaged_result: np.ndarray = None

    def __iter__(self):
        self.x = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.D = 0
        self.cum_y = np.zeros_like(self.x)
        self.averaged_result = None

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
        self.averaged_result = self.cum_y / self.iter

        if self.projection_type == ProjectionType.BREGMAN:
            self.D = np.linalg.norm(self.y - self.x, 1)
        else:
            self.D = np.linalg.norm(self.y - self.x)

        if self.D >= self.eps or self.iter < self.min_iters:
            self.px = self.x

            if self.projection_type == ProjectionType.BREGMAN:
                self.x[:self.problem.m] = np.exp(np.log(self.y[:self.problem.m]) - self.lam * (self.problem.A(self.y)[:self.problem.m] - Ax[:self.problem.m]))
                self.x[self.problem.m:] = self.y[self.problem.m:] - self.lam * (self.problem.A(self.y)[self.problem.m:] - Ax[self.problem.m:])
                # self.x = np.exp(np.log(self.y) - self.lam * (self.problem.A(self.y) - Ax))
                # self.x = np.exp(np.log(self.y) - self.lam * (self.problem.A(self.y) - Ax))

                # self.x = self.y * np.exp(-self.lam * (self.problem.A(self.y) - Ax))

                # PageRank optimized :-)
                # self.x = self.y * np.exp(-self.lam * (self.problem.A(self.y) - Ax) - 1.)
            else:
                self.x = self.y - self.lam * (self.problem.A(self.y) - Ax)

            self.operator_count += 1

    def doPostStep(self):
        if self.iter > 0:
            val_for_gap = self.averaged_result
        else:  # calc gap from x0
            val_for_gap = self.x

        # t = self.problem.F(val_for_gap)
        self.setHistoryData(x=self.x, y=val_for_gap, step_delta_norm=self.D,
                            goal_func_value=self.problem.F(self.y), goal_func_from_average=self.problem.F(val_for_gap))

    def isStopConditionMet(self):
        stop_condition_met = False
        if self.stop_condition == StopCondition.STEP_SIZE:
            stop_condition_met = (self.D < self.eps)
        elif self.stop_condition == StopCondition.GAP:
            stop_condition_met = (self.iter > 0 and self.problem.F(self.averaged_result) < self.eps)
        elif self.stop_condition == StopCondition.EXACT_SOL_DIST:
            stop_condition_met = (np.linalg.norm(self.x - self.problem.xtest) < self.eps)

        return super(Tseng, self).isStopConditionMet() or stop_condition_met

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
