import numpy as np

from methods.IterGradTypeMethod import IterGradTypeMethod
from problems.viproblem import VIProblem


class MalitskyTamAdaptive(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 x1: np.ndarray, lam1: float = 0.1, tau: float = 0.25,
                 min_iters: int = 0, max_iters=5000, hr_name: str = None):
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters, hr_name=hr_name)

        self.ppx = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.x1 = x1.copy()

        self.Ax = None
        self.Apx = None
        self.delta_Ax = None

        self.p_lam = self.lam0 = lam
        self.lam1 = lam1

        self.tau = tau

        self.D = 0
        self.D_1 = 0

    def __iter__(self):
        self.ppx = self.problem.x0.copy()
        self.px = self.problem.x0.copy()
        self.x = self.x1.copy()

        self.p_lam = self.lam0
        self.lam = self.lam1

        self.Apx = self.problem.A(self.px)
        self.Ax = self.problem.A(self.x)
        self.delta_Ax = self.Ax - self.Apx

        self.D = np.linalg.norm(self.x - self.px)
        self.D_1 = 0

        return super().__iter__()

    def doStep(self):
        self.ppx = self.px
        self.px = self.x.copy()
        self.x = self.problem.Project(self.x - self.lam * self.Ax - self.p_lam * self.delta_Ax)
        self.projections_count += 1

        self.Apx = self.Ax
        self.Ax = self.problem.A(self.x)
        self.delta_Ax = self.Ax - self.Apx

        self.operator_count += 1

        self.D_1 = self.D
        self.D = np.linalg.norm(self.x - self.px)

        if self.D_1 > self.zero_delta and self.D > self.zero_delta:
            self.p_lam = self.lam

            nr = np.linalg.norm(self.delta_Ax)
            if nr > self.zero_delta:
                t = self.tau * self.D / nr
                if self.lam >= t:
                    self.lam = t

    def doPostStep(self):
        self.setHistoryData(x=self.x, step_delta_norm=self.D, goal_func_value=self.problem.F(self.x))

    def isStopConditionMet(self):
        return super(MalitskyTamAdaptive, self).isStopConditionMet() or (self.D < self.eps and self.D_1 < self.eps)

    def __next__(self):
        return super(MalitskyTamAdaptive, self).__next__()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}".format(self.problem.XToString(self.problem.x0))

    def currentState(self) -> dict:
        # return dict(super().currentState(), x=([*self.x, self.problem.F(self.x)],
        # [*self.y, self.problem.F(self.y)]), lam=self.lam)

        return dict(super().currentState(), x=(self.x,), F=(self.problem.F(self.x),),
                    D=self.D, lam=self.lam, iterEndTime=self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam,
                                                         self.problem.FValToString(self.problem.F(self.x)))
