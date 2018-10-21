import time
from scipy import linalg
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod

class KorpelevichMod(IterGradTypeMethod):

    def __init__(self, problem:VIProblem, eps: float = 0.0001, lam:float = 0.1, maxLam:float = 0.9, *, min_iters:int = 0):
        super().__init__(problem, eps, lam, min_iters=min_iters)
        self.maxLam = maxLam
        self.x = self.px = self.y = self.problem.x0.copy()
        self.D = 0

    def __iter__(self):
        self.x = self.px = self.problem.x0.copy()
        return super().__iter__()

    def __next__(self):
        self.D = linalg.norm(self.x - self.y)
        if self.D >= self.eps or self.iter == 0 or self.min_iters > self.iter:
            self.iter+=1
            self.y = self.problem.Project(self.x - self.lam * self.problem.GradF(self.x))

            if self.iter > 1 and linalg.norm(self.x - self.y) >= self.eps:
                t = 0.9 * linalg.norm(self.x - self.y) / linalg.norm(self.problem.GradF(self.x) - self.problem.GradF(self.y))
                self.lam = t if t <= self.maxLam else self.maxLam
            else:
                self.lam = self.maxLam

            self.px, self.x = self.x, self.problem.Project(self.x - self.lam * self.problem.GradF(self.y));

            self.iterEndTime = time.process_time()

            return self.currentState()
        else:
            raise StopIteration()

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}; MaxLam: {1}".format(self.problem.XToString(self.problem.x0), self.maxLam)

    def currentState(self) -> dict:
        # return dict(super().currentState(), x=([*self.x, self.problem.F(self.x)], [*self.y, self.problem.F(self.y)]), lam=self.lam)
        return dict(super().currentState(), x=(self.x, self.y), F=(self.problem.F(self.x), self.problem.F(self.y)),
                    D=self.D, lam=self.lam, iterEndTime = self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: x: {1}; lam: {2}; F(x): {3}".format(self.iter, self.problem.XToString(self.x), self.lam, self.problem.FValToString(self.problem.F(self.x)))

