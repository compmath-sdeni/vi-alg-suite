import time
from typing import Union

import numpy as np
from scipy import linalg

from methods.algorithm_params import StopCondition
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class MalitskyTam(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, x1: np.ndarray,
                 min_iters: int = 0, max_iters=5000, hr_name: str = None,
                 projection_type: ProjectionType = ProjectionType.EUCLID,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type, stop_condition=stop_condition)

        self.ppx = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.x = x1.copy()
        self.x1 = self.x.copy()

        self.cum_x = np.zeros_like(self.x)
        self.averaged_result: np.ndarray = None

        self.Apx = self.problem.A(self.px)
        self.Ax = self.problem.A(self.x)

        self.D: float = 0
        self.D2: float = 0

    def __iter__(self):
        self.ppx = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.x = self.x1.copy()
        # self.cum_x = self.x
        self.averaged_result = None

        self.D = np.linalg.norm(self.x - self.px)
        self.D2 = 0

        self.Apx = self.problem.A(self.px)
        self.Ax = self.problem.A(self.x)
        self.operator_count += 2

        return super().__iter__()

    def doStep(self):
        self.ppx = self.px
        self.px = self.x

        if self.projection_type == ProjectionType.BREGMAN:
            self.x = self.problem.bregmanProject(self.x, - self.lam * self.Ax - self.lam * (self.Ax - self.Apx))
        else:
            self.x = self.problem.Project(self.x - self.lam * self.Ax - self.lam * (self.Ax - self.Apx))

        self.projections_count += 1

        self.cum_x += self.x
        self.averaged_result = self.cum_x / self.iter

        self.Apx = self.Ax
        self.Ax = self.problem.A(self.x)
        self.operator_count += 1

        if self.projection_type == ProjectionType.BREGMAN:
            self.D = np.linalg.norm(self.x - self.px, 1)
            self.D2 = np.linalg.norm(self.px - self.ppx, 1)
        else:
            self.D = np.linalg.norm(self.x - self.px)
            self.D2 = np.linalg.norm(self.px - self.ppx)

    def doPostStep(self):
        if self.iter > 0:
            val_for_gap = self.averaged_result
        else:  # calc gap from x0
            val_for_gap = self.x

        self.setHistoryData(x=self.x, y=val_for_gap, step_delta_norm=self.D + self.D2,
                            goal_func_value=self.problem.F(self.x), goal_func_from_average=self.problem.F(val_for_gap))

    def isStopConditionMet(self):
        stop_condition_met: bool = False
        if self.stop_condition == StopCondition.STEP_SIZE:
            stop_condition_met = (self.D + self.D2 < self.eps)
        elif self.stop_condition == StopCondition.GAP:
            stop_condition_met = (self.iter > 0 and self.problem.F(self.averaged_result) < self.eps)
        elif self.stop_condition == StopCondition.EXACT_SOL_DIST:
            stop_condition_met = (np.linalg.norm(self.x - self.problem.xtest) < self.eps)

        return super(MalitskyTam, self).isStopConditionMet() or stop_condition_met

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
