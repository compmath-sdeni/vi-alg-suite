import numpy as np

from methods.algorithm_params import StopCondition
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class ExtrapolationFromPast(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, y0: np.ndarray,
                 min_iters: int = 0, max_iters=5000,
                 hr_name: str = None, projection_type: ProjectionType = ProjectionType.EUCLID,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type, stop_condition=stop_condition)

        self.x0 = self.problem.x0
        self.y0 = y0
        self.y = self.y0

        self.x: np.ndarray = self.problem.x0
        self.Ay: np.ndarray = self.problem.A(self.y0)

        self.cum_y = np.zeros_like(self.x)
        self.averaged_result: np.ndarray = None

        self.D: float = 0
        self.D2: float = 0

    def __iter__(self):
        self.x = self.x0.copy()
        self.y = self.y0.copy()

        self.projections_count = 0
        self.operator_count = 0

        self.Ay = self.problem.A(self.y)
        self.operator_count += 1

        self.D = 0
        self.D2 = 0

        self.cum_y = np.zeros_like(self.y)
        self.averaged_result = None

        return super().__iter__()

    def doStep(self):

        if self.projection_type == ProjectionType.BREGMAN:
            self.y = self.problem.bregmanProject(self.x, - self.lam * self.Ay)
        else:
            self.y = self.problem.Project(self.x - self.lam * self.Ay)

        self.cum_y += self.y
        self.averaged_result = self.cum_y / self.iter

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

    def doPostStep(self):
        if self.iter > 0:
            val_for_gap = self.averaged_result
        else:  # calc gap from y0
            val_for_gap = self.y

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

        return super().isStopConditionMet() or stop_condition_met

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
