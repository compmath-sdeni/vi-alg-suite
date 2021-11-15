import time
from typing import Union

import numpy as np
from scipy import linalg

from methods.malitsky_tam import MalitskyTam
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod


class MalitskyTamBregman(MalitskyTam):

    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *, x1: np.ndarray,
                 min_iters: int = 0, max_iters=5000, hr_name: str = None):
        super().__init__(problem, eps, lam, x1=x1, min_iters=min_iters, max_iters=max_iters, hr_name=hr_name)

    def __iter__(self):
        return super().__iter__()

    def euclidProj(self, x: np.ndarray, a: np.ndarray):
        return self.problem.Project(x + a)

    def doStep(self):
        self.ppx = self.px
        self.px = self.x
        self.x = self.problem.bregmanProject(self.x, - (self.lam * self.Ax + self.lam * (self.Ax - self.Apx)))

        self.cum_x += self.x

        self.projections_count += 1

        self.Apx = self.Ax
        self.Ax = self.problem.A(self.x)
        self.operator_count += 1

        self.D = np.linalg.norm(self.x - self.px)
        self.D2 = np.linalg.norm(self.px - self.ppx)
