import numpy as np
from numpy import inf

from methods.algorithm_params import StopCondition
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class ExtrapolationFromPastAdapt(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, y0: np.ndarray,
                 min_iters: int = 0, max_iters=5000, tau: float = 0.3,
                 hr_name: str = None, projection_type: ProjectionType = ProjectionType.EUCLID,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type, stop_condition=stop_condition)

        self.x0 = self.problem.x0
        self.y0 = y0
        self.y = self.y0

        self.x: np.ndarray = self.problem.x0
        self.Ay: np.ndarray = self.problem.A(self.y0)

        self.cum_y: np.ndarray = np.zeros_like(self.y)

        self.D: float = 0
        self.D2: float = 0

        self.lam0 = lam
        self.tau = tau

        self.hist_for_avg: np.ndarray = None
        self.x_min_gap: np.ndarray = self.x

    def __iter__(self):
        self.x = self.x0.copy()
        self.y = self.y0.copy()
        self.x_min_gap: np.ndarray = self.x

        self.projections_count = 0
        self.operator_count = 0

        self.Ay = self.problem.A(self.y)
        self.operator_count += 1

        self.D = 0
        self.D2 = 0

        self.lam = self.lam0

        # self.cum_y = self.y # start average from y0
        self.cum_y = np.zeros_like(self.y) # start average from y1

        # self.hist_for_avg = np.zeros((self.max_iters + 1, self.y.shape[0]))
        # self.hist_for_avg[self.iter] = self.y

        return super().__iter__()

    def doStep(self):
        py = self.y
        if self.projection_type == ProjectionType.BREGMAN:
            self.y = self.problem.bregmanProject(self.x, - self.lam * self.Ay)
        else:
            self.y = self.problem.Project(self.x - self.lam * self.Ay)

        self.cum_y += self.y

        # the first time we get here, iter = 1. So, to start average from y1 and not from y0, we need iter-1
#        self.hist_for_avg[self.iter-1] = self.y

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

        if self.D + self.D2 >= self.zero_delta:
            if self.projection_type.BREGMAN:
                delta_A = np.linalg.norm(self.Ay - pAy, inf)
            else:
                delta_A = np.linalg.norm(self.Ay - pAy)

            if delta_A > self.zero_delta:
                if self.projection_type == ProjectionType.BREGMAN:
                    difnorm = np.linalg.norm(self.y - py, 1)
                else:
                    difnorm = np.linalg.norm(self.y - py)

                t = self.tau * difnorm / delta_A
                if self.lam > t:
                    self.lam = t

    def doPostStep(self):
        # if we want to start from y0, we need to use iter + 1 and do not skip the first time
        # otherwise, we need to use iter and skip iteration-0
        # val_for_gap = self.cum_y / (self.iter + 1)
        if self.iter > 0:
            val_for_gap = self.cum_y / self.iter
            # start_iter_for_sum = 0  # int((self.iter) / 2)
            # t = self.hist_for_avg[start_iter_for_sum: self.iter]
            # d = float(self.iter - start_iter_for_sum)
            # val_for_gap2 = t.sum(axis=0) / d
        else:  # calc gap from y0
            val_for_gap = self.y
#            val_for_gap2 = val_for_gap

        # if self.problem.F(self.x_min_gap) > self.problem.F(self.y):
        #     self.x_min_gap = self.y

        self.setHistoryData(x=self.x, y=val_for_gap, step_delta_norm=self.D + self.D2,
                            goal_func_value=self.problem.F(self.x), goal_func_from_average=self.problem.F(val_for_gap))

    def isStopConditionMet(self):
        stop_condition_met: bool = False
        if self.stop_condition == StopCondition.STEP_SIZE:
            stop_condition_met = (self.D + self.D2 < self.eps)
        elif self.stop_condition == StopCondition.GAP:
            if self.iter > 0:
                val_for_gap = self.cum_y / self.iter
            else:  # calc gap from y0
                val_for_gap = self.y
            stop_condition_met = (self.problem.F(val_for_gap) < self.eps)
        elif self.stop_condition == StopCondition.EXACT_SOL_DIST:
            stop_condition_met = (np.linalg.norm(self.x - self.problem.xtest) < self.eps)

        return super().isStopConditionMet() or stop_condition_met

    def __next__(self):
        return super().__next__()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; y0: {0}; ".format(self.problem.XToString(self.y0))

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x,), F=(self.problem.F(self.x),),
                    D=self.D + self.D2, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))
