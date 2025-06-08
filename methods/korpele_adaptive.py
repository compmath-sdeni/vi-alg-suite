import numpy as np
from numpy import inf

from methods.algorithm_params import StopCondition
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod, ProjectionType


class KorpelevichExtragradAdapt(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, y0: np.ndarray,
                 min_iters: int = 0, max_iters=5000, tau: float = 0.3,
                 hr_name: str = None, projection_type: ProjectionType = ProjectionType.EUCLID,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE,
                 use_step_increase: bool = False, step_increase_seq_rule = None):

        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters,
                         hr_name=hr_name, projection_type=projection_type, stop_condition=stop_condition
                         )

        self.x0 = self.problem.x0
        self.y0 = self.problem.x0
        self.y = self.y0

        self.x: np.ndarray = self.x0
        self.Ax: np.ndarray = self.problem.A(self.x)
        self.Ay: np.ndarray = self.problem.A(self.y)

        self.cum_x = self.x.copy()
        self.averaged_result: np.ndarray = None

        self.D: float = 0

        self.lam0 = lam
        self.tau = tau

        self.hist_for_avg: np.ndarray = None
        self.x_min_gap: np.ndarray = self.x

        self.use_step_increase = use_step_increase
        self.step_increase_seq_rule = step_increase_seq_rule

    def __iter__(self):
        self.x = self.x0.copy()
        self.y = self.y0.copy()
        self.x_min_gap: np.ndarray = self.x

        self.projections_count = 0
        self.operator_count = 0

        self.Ax = self.problem.A(self.x)
        self.Ay = self.problem.A(self.y)
        self.operator_count += 1

        self.D = 0

        self.lam = self.lam0

        self.cum_x = self.x.copy()
        self.averaged_result = None

        return super().__iter__()

    def doStep(self):
        py = self.y #.copy()
        self.Ax = self.problem.A(self.x)
        if self.projection_type == ProjectionType.BREGMAN:
            self.y = self.problem.bregmanProject(self.x, - self.lam * self.Ax)
        else:
            self.y = self.problem.Project(self.x - self.lam * self.Ax)

        pAy = self.Ay #.copy()
        self.Ay = self.problem.A(self.y)

        px = self.x #.copy()
        if self.projection_type == ProjectionType.BREGMAN:
            self.x = self.problem.bregmanProject(self.x, - self.lam * self.Ay)
        else:
            self.x = self.problem.Project(self.x - self.lam * self.Ay)

        # the first time we get here, iter = 1, and x0 is already in cum_x.
        self.cum_x += self.x
        self.averaged_result = self.cum_x / (self.iter + 1)

        if self.projection_type == ProjectionType.BREGMAN:
            self.D = np.linalg.norm(px - self.y, 1)
        else:
            self.D = np.linalg.norm(px - self.y)

        self.projections_count += 2
        self.operator_count += 2

        if self.D >= self.zero_delta:
            diff_A = self.Ax - self.Ay
            diff_pxy = px - self.y

            if self.use_step_increase:
                diff_newx_y = self.x - self.y
                if self.projection_type == ProjectionType.BREGMAN:
                    difnorm = np.linalg.norm(diff_pxy, inf)
                    dif_newx_norm = np.linalg.norm(diff_newx_y, inf)
                    delta_A = np.linalg.norm(diff_A, inf)
                else:
                    difnorm = np.linalg.norm(diff_pxy, 2)
                    dif_newx_norm = np.linalg.norm(diff_newx_y, 2)
                    delta_A = np.linalg.norm(diff_A, 2)

                if self.projection_type == ProjectionType.BREGMAN:
                    dot_prod_to_check = 1.0 # for bregman div always check!
                else:
                    dot_prod_to_check = np.dot(diff_A, diff_newx_y)

                lam_inc = self.step_increase_seq_rule(self.iter)
                self.lam += lam_inc
                if dot_prod_to_check > 0:
                    # t = self.tau * 0.5 * (difnorm + dif_newx_norm) / dot_prod_to_check
                    t = self.tau * difnorm / delta_A
                    if self.lam > t:
                        self.lam = t
            else:
                if self.projection_type == ProjectionType.BREGMAN:
                    delta_A = np.linalg.norm(diff_A, inf)
                else:
                    delta_A = np.linalg.norm(diff_A, 2)

                if delta_A > self.zero_delta:
                    if self.projection_type == ProjectionType.BREGMAN:
                        difnorm = np.linalg.norm(diff_pxy, inf)
                    else:
                        difnorm = np.linalg.norm(diff_pxy, 2)

                    t = self.tau * difnorm / delta_A
                    if self.lam > t:
                        self.lam = t

    def doPostStep(self):
        # if we want to start from y0, we need to use iter + 1 and do not skip the first time
        # otherwise, we need to use iter and skip iteration-0
        # val_for_gap = self.cum_y / (self.iter + 1)
        if self.iter > 0:
            val_for_gap = self.averaged_result
            # start_iter_for_sum = 0  # int((self.iter) / 2)
            # t = self.hist_for_avg[start_iter_for_sum: self.iter]
            # d = float(self.iter - start_iter_for_sum)
            # val_for_gap2 = t.sum(axis=0) / d
        else:  # calc gap from y0
            val_for_gap = self.x
        #            val_for_gap2 = val_for_gap

        # if self.problem.F(self.x_min_gap) > self.problem.F(self.y):
        #     self.x_min_gap = self.y

        self.setHistoryData(x=self.x, y=val_for_gap, step_delta_norm=self.D,
                            goal_func_value=self.problem.F(self.x), goal_func_from_average=self.problem.F(val_for_gap))

    def isStopConditionMet(self):
        stop_condition_met: bool = False
        if self.stop_condition == StopCondition.STEP_SIZE:
            stop_condition_met = (self.D < self.eps)
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
        return dict(super().currentState(), x=(self.x,), F=(self.problem.F(self.x),),
                    D=self.D, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))

    def isAdaptive(self) -> bool:
        return True
