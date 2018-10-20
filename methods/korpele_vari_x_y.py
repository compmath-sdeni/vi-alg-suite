from typing import Union

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod
import numpy as np
from scipy import linalg

from utils.print_utils import *


class KorpeleVariX_Y(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters = 0, phi: float = 0.75, gap: int = 10):

        super().__init__(problem, eps, lam, min_iters=min_iters)
        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.px: Union[np.ndarray, float] = self.x.copy()
        self.D: float = 0
        self.y: Union[np.ndarray, float] = self.x.copy()
        self.initialLam = self.lam = lam
        self.phi = phi
        self.gap = gap

        self.ax = None
        self.ay = None
        self.fixedLambdaSteps = 0

    def __iter__(self):
        self.x = self.problem.x0.copy()
        self.px = self.x.copy()
        self.D: float = 0
        return super().__iter__()

    def calcLam(self):
        norm = np.linalg.norm(self.ax - self.ay)
        increased = False

        if self.gap>=0 and self.fixedLambdaSteps>self.gap:
            self.lam *= 1.05
            increased = True
            self.fixedLambdaSteps = 0

        if norm > self.eps:
            l = self.phi * np.linalg.norm(self.px - self.y)/norm
            if l<self.lam:
                self.lam = l
                increased = False
                self.fixedLambdaSteps = 0
            else:
                self.fixedLambdaSteps += 1
        else:
            self.fixedLambdaSteps += 1

        if increased:
            print('!! Lam increased to ', self.lam)

    def __next__(self):
        #self.D = linalg.norm(self.x - self.px)
        self.D = linalg.norm(self.x - self.y)
        if self.min_iters > self.iter or self.D >= self.eps or self.iter == 0:
            self.iter += 1

            if self.iter != 1:
                self.calcLam()

            self.ax = self.problem.GradF(self.x)
            self.y: Union[np.ndarray, float]  = self.problem.Project(self.x - self.lam * self.ax)

            self.ay = self.problem.GradF(self.y)
            self.px, self.x = self.x, self.problem.Project(self.x - self.lam * self.ay)

            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.px, self.y), D=self.D,
                    F=(self.problem.F(self.px), self.problem.F(self.y)), lam=self.lam)

    def currentStateString(self) -> str:
        return "{0}: D: {1}; lam:{2}; x: {3}; y: {4}; F: {5}".format(
            self.iter, self.D, scalarToString(self.lam), self.problem.XToString(self.px),
            self.problem.XToString(self.y), self.problem.FValToString(self.problem.F(self.px)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString()+"; x0: {0}".format(self.problem.XToString(self.problem.x0))
