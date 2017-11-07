from typing import Union
from scipy import linalg
import numpy as np

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod


class PopovSubgrad(IterGradTypeMethod):
    def __init__(self, problem:VIProblem, eps: float = 0.0001, lam:float = 0.1, *, min_iters:int = 0):
        super().__init__(problem, eps, lam, min_iters=min_iters)

        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.y: Union[np.ndarray, float] = self.x.copy()

        self.px: Union[np.ndarray, float] = self.x.copy()
        self.py: Union[np.ndarray, float] = self.x.copy()
        self.ppy: Union[np.ndarray, float] = self.x.copy()

        self.D: float = 0

    def _supportHProject(self, x):
        #project to Hn = (c,x)<=b
        c = self.x - self.lam*self.problem.GradF(self.py) - self.y
        b = np.dot(c,self.y)

        if np.dot(c,x) > b:
            # Px = x + (b - <c,x>)*c/<c,c>
            return x + (b - np.dot(c,x))*c/(np.dot(c,c))
        else:
            return x


    def __iter__(self):
        self.x = self.px = self.problem.x0.copy()
        self.y = self.py = self.ppy = self.problem.x0.copy()

        self.px, self.x = self.x, self.problem.Project(self.x - self.lam * self.problem.GradF(self.ppy))
        self.ppy, self.py = self.py, self.problem.Project(self.x - self.lam * self.problem.GradF(self.ppy))

        return super().__iter__()

    def __next__(self):
        self.D = linalg.norm(self.x - self.px)
        if self.min_iters > self.iter or self.D >= self.eps or (linalg.norm(self.py - self.ppy) >= self.eps or linalg.norm(self.y - self.py) >= self.eps) or self.iter == 0:
            self.iter+=1

            self.px, self.x = self.x, self._supportHProject(self.x - self.lam * self.problem.GradF(self.y))

            self.ppy, self.py, self.y = self.py, self.y, self.problem.Project(self.x - self.lam * self.problem.GradF(self.y))

            return self.currentState()
        else:
            raise StopIteration()


    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}; y0: {1}".format(self.problem.XToString(self.problem.x0), self.problem.XToString(self.problem.x0))

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.py, self.y, self.x), F=(self.problem.F(self.py), self.problem.F(self.y), self.problem.F(self.x)),
                    D=self.D)

    def currentStateString(self) -> str:
        return "{0}: D: {1}; x: {2}; y: {3}; F: {4}".format(self.iter, self.D, self.problem.XToString(self.x), self.problem.XToString(self.y), self.problem.FValToString(self.problem.F(self.x)))