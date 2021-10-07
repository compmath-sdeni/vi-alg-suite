import numpy as np
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod


class ExtrapolationFromPastAdapt(IterGradTypeMethod):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 y0: np.ndarray, tau: float = 0.3,
                 min_iters: int = 0, max_iters=5000, hr_name: str = None):
        super().__init__(problem, eps, lam, min_iters=min_iters, max_iters=max_iters, hr_name=hr_name)

        self.x0 = self.problem.x0
        self.y0 = y0

        self.lam0 = lam

        self.tau = tau

        self.x: np.ndarray = self.problem.x0
        self.y: np.ndarray = self.y0
        self.Ay: np.ndarray = self.problem.A(self.y)

        self.D: float = 0
        self.D2: float = 0

    def __iter__(self):
        self.x = self.x0.copy()
        self.y = self.y0.copy()

        self.projections_count = 0
        self.operator_count = 0

        self.lam = self.lam0

        self.Ay = self.problem.A(self.y)
        self.operator_count += 1

        self.D = 0
        self.D2 = 0

        return super().__iter__()

    def doStep(self):
        py = self.y
        self.y = self.problem.Project(self.x - self.lam * self.Ay)

        pAy = self.Ay
        self.Ay = self.problem.A(self.y)

        px = self.x
        self.x = self.problem.Project(self.x - self.lam * self.Ay)

        self.D = np.linalg.norm(px - self.y)
        self.D2 = np.linalg.norm(self.x - px)

        self.projections_count += 2
        self.operator_count += 1

        if self.D + self.D2 >= self.eps:
            delta_A = np.linalg.norm(pAy - self.Ay)
            if delta_A > self.zero_delta:
                t = self.tau * np.linalg.norm(py - self.y) / delta_A
                if self.lam > t:
                    self.lam = t

    def doPostStep(self):
        self.setHistoryData(x=self.x, step_delta_norm=self.D + self.D2, goal_func_value=self.problem.F(self.x))

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
