import numpy as np
from numpy import inf

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class ExtrapolationFromPastAdapt(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, y0: np.ndarray,
                 min_iters: int = 0, max_iters=5000, tau: float = 0.3,
                 hr_name: str = None, projection_type: ProjectionType = ProjectionType.EUCLID):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type)

        self.x0 = self.problem.x0
        self.y0 = y0
        self.y = self.y0

        self.x: np.ndarray = self.problem.x0
        self.Ay: np.ndarray = self.problem.A(self.y0)

        self.cum_y = np.zeros_like(self.x)

        self.D: float = 0
        self.D2: float = 0

        self.lam0 = lam
        self.tau = tau


    def __iter__(self):
        self.x = self.x0.copy()

        self.projections_count = 0
        self.operator_count = 0

        self.Ay = self.problem.A(self.y0)
        self.operator_count += 1

        self.D = 0
        self.D2 = 0

        self.lam = self.lam0

        return super().__iter__()

    def doStep(self):
        py = self.y
        if self.projection_type == ProjectionType.BREGMAN:
            self.y = self.problem.bregmanProject(self.x, - self.lam * self.Ay)
        else:
            self.y = self.problem.Project(self.x - self.lam * self.Ay)

        self.cum_y += self.y
        pAy = self.Ay
        self.Ay = self.problem.A(self.y)
        px = self.x

        if self.projection_type == ProjectionType.BREGMAN:
            self.x = self.problem.bregmanProject(self.x, - self.lam * self.Ay)
        else:
            self.x = self.problem.Project(self.x - self.lam * self.Ay)

        if self.projection_type == ProjectionType.BREGMAN:
            self.D = np.linalg.norm(px - self.y, 1)
            self.D2 = np.linalg.norm(self.x - px, 1)
        else:
            self.D = np.linalg.norm(px - self.y)
            self.D2 = np.linalg.norm(self.x - px)

        self.projections_count += 2
        self.operator_count += 1

        if self.D + self.D2 >= self.eps:
            if self.projection_type.BREGMAN:
                delta_A = np.linalg.norm(pAy - self.Ay, inf)
            else:
                delta_A = np.linalg.norm(pAy - self.Ay)

            if delta_A > self.zero_delta:
                if self.projection_type.BREGMAN:
                    difnorm = np.linalg.norm(py - self.y, 1)
                else:
                    difnorm = np.linalg.norm(py - self.y)

                t = self.tau * difnorm / delta_A
                if self.lam > t:
                    self.lam = t

    def doPostStep(self):
        val_for_gap = self.cum_y / (self.iter + 1)
        self.setHistoryData(x=self.x, y=val_for_gap, step_delta_norm=self.D + self.D2,
                            goal_func_value=self.problem.F(self.x), goal_func_from_average=self.problem.F(val_for_gap))

    def isStopConditionMet(self):
        return super().isStopConditionMet() or (self.D + self.D2 < self.eps)

    def __next__(self):
        return super().__next__()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; y0: {0}; ".format(self.problem.XToString(self.y0))

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, ), F=(self.problem.F(self.x), ),
                    D=self.D + self.D2, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))
