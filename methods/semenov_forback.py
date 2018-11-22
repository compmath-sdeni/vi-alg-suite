from typing import Union
import time

from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod
import numpy as np
from scipy import linalg

from utils.print_utils import *


class SemenovForBack(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters = 0):

        super().__init__(problem, eps, lam, min_iters=min_iters)
        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.px: Union[np.ndarray, float] = self.x.copy()
        self.D: float = 0

        self.ax: np.ndarray = None
        self.pax: np.ndarray = None

    def __iter__(self):
        self.px = self.problem.x0.copy()
        self.pax = self.problem.GradF(self.px)
        self.x = self.problem.Project(self.px - self.lam * self.pax)
        self.D: float = 0
        return super().__iter__()

    def __next__(self):
        self.D = linalg.norm(self.x - self.px)
        if self.min_iters > self.iter or self.D >= self.eps or self.iter == 0:
            self.iter += 1

            self.ax = self.problem.GradF(self.x)
            self.px, self.x = self.x, self.problem.Project(self.x - self.lam * 2 * self.ax + self.lam * self.pax)
            self.pax = np.copy(self.ax)

            self.iterEndTime = time.process_time()

            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.px, self.px), D=(self.D, linalg.norm(self.x - self.px)),
                    F=(self.problem.F(self.px),), lam=self.lam, iterEndTime = self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0}: D: {1}; lam:{2}; x: {3}; F: {4}".format(
            self.iter, self.D, scalarToString(self.lam), self.problem.XToString(self.px),
            self.problem.FValToString(self.problem.F(self.px)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString()+"; x0: {0}".format(self.problem.XToString(self.problem.x0))
