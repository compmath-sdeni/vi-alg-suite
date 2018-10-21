from typing import Union
import time

import numpy as np
from scipy import linalg

from methods.IterGradTypeMethod import IterGradTypeMethod
from problems.splittable_vi_problem import SplittableVIProblem


class BatchedGradProj(IterGradTypeMethod):
    def __init__(self, problem: SplittableVIProblem, eps: float = 0.0001, lam_init: float = 0.5, *, min_iters: int = 0, split_count: int = None, hr_name:str = None):
        super().__init__(problem, eps, lam_init, min_iters=min_iters, hr_name=hr_name)
        self.problem: SplittableVIProblem = problem
        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.px: Union[np.ndarray, float] = self.x.copy()
        self.D: float = None
        self.mini_iter: int = 0

        self.lam_init: float = lam_init
        if self.problem.lam_override is not None:
            self.lam_init = self.problem.lam_override

        self.topsum: float = 0
        self.bottomsum: float = 0
        self.z: float = self.x

        self.split_count = split_count

    def __iter__(self):
        self.x = self.problem.x0.copy()
        self.px = self.x.copy()
        self.D = None
        self.mini_iter = 0
        self.z = self.x

        if self.split_count is not None:
            self.problem.set_split(self.split_count)

        return super().__iter__()

    def do_inner_iterations(self):
        for i in range(self.problem.K):
            self.x = self.problem.Project(self.x - self.lam * self.problem.GradFi(self.x, i))
            # print("TempX: {0}", self.problem.XToString(self.x))
            self.mini_iter += 1

    def __next__(self) -> dict:
        if self.min_iters > self.iter or self.iter == 0 or self.D >= self.eps:
            self.iter += 1
            self.px = self.x.copy()

            #self.lam = self.lam_init
            self.lam = self.lam_init / (self.iter ** 0.25)
            #self.lam = self.lam_init / self.iter

            self.do_inner_iterations()
            self.D = linalg.norm(self.x - self.px)
            self.topsum += self.lam * self.x
            self.bottomsum += self.lam
            self.z = self.topsum/self.bottomsum

            self.iterEndTime = time.process_time()

            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, self.z), D=self.D, F=(self.problem.F(self.x), self.problem.F(self.z)),
                    mini_iter=self.mini_iter, iterEndTime = self.iterEndTime)

    def currentStateString(self) -> str:
        return "{0} ({1}): D: {2}; x: {3}; z: {4}; F(x): {5}; F(z): {6}".format(
            self.iter, self.mini_iter, self.D, self.problem.XToString(self.x), self.problem.XToString(self.z),
            self.problem.FValToString(self.problem.F(self.x)), self.problem.FValToString(self.problem.F(self.z)))

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0}".format(self.problem.XToString(self.problem.x0))
