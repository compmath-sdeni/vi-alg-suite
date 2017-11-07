from typing import Union, Callable
import numpy as np
from scipy import linalg
from problems.viproblem import VIProblem
from methods.IterGradTypeMethod import IterGradTypeMethod

from utils.print_utils import *


class VaristepThree(IterGradTypeMethod):
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, sigma: float = 1, theta: float = 0.9,
                 tau: float = 0.5, stab: int = 5, *, min_iters: int = 0, xstar: np.ndarray = None,
                 alfa_calc: Callable = None):
        super().__init__(problem, eps, lam, min_iters=min_iters)
        self.theta: float = theta
        self.sigma: float = sigma
        self.tau: float = tau
        self.alfaCalc = alfa_calc if alfa_calc is not None else lambda i: 1.0 / np.sqrt(i + 1)

        self._currentStateInfo: dict = {'x': 0, 'y': 0, 'lam': 0}
        self.j: int = 0

        self.stab: int = stab
        self.samej: int = self.stab  # we need to calc j the first time

        self.x: Union[np.ndarray, float] = self.problem.x0.copy()
        self.y: Union[np.ndarray, float] = self.x.copy()
        self.D: float = 0
        self.xstar: Union[np.ndarray, float] = xstar if xstar is not None else self.problem.x0.copy()

    def _supportHProject(self, x: np.ndarray):
        c = self.x - self.lam * self.problem.GradF(self.x) - self.y
        b = np.dot(c, self.y)

        if np.dot(c, x) > b:
            return x + (b - np.dot(c, x)) * c / (np.dot(c, c))
        else:
            return x

    def calcLam(self):
        # self.lam = 0.2
        # self.j = 10
        # return

        j = 0

        # if self.samej < self.stab:
        #     self.samej += 1
        #     return
        #
        # self.samej = 0
        #
        # if self.stab >= 0:
        #     j = self.j
        #     if j > 0:
        #         j -= 1
        # else:
        #     j = 0

        x = self.x
        sgm = self.sigma
        tet = self.theta
        tau = self.tau
        tauj = tau ** j
        F = self.problem.GradF
        Pc = self.problem.Project

        Fx = F(x)

        vin = Pc(x - sgm * tauj * Fx)
        vl = F(vin) - Fx
        vr = vin - x

        lp = sgm * tauj * np.sqrt(np.dot(vl, vl))
        rp = tet * np.sqrt(np.dot(vr, vr))

        # print("\t\t\t\tlp: {0}, rp: {1}, vl: {2}".format(lp, rp, vl))

        while (lp > rp):
            tauj *= tau
            j = j + 1

            vin = Pc(x - sgm * tauj * Fx)
            vl = F(vin) - Fx
            vr = vin - x

            lp = sgm * tauj * np.sqrt(np.dot(vl, vl))
            rp = tet * np.sqrt(np.dot(vr, vr))

        self.lam = sgm * tauj
        self.j = j

    def projectLast(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        hi = np.dot(x - y, y - z)
        mu = np.dot(x - y, x - y)
        nu = np.dot(y - z, y - z)
        ro = mu * nu - hi ** 2

        if abs(ro) < 0.0000000001 and hi >= 0:
            return z
        elif ro > 0 and hi * nu >= ro:
            return x + (1.0 + hi / nu) * (z - y)
        elif ro > 0 and hi * nu < ro:
            return y + nu / ro * (hi * (x - y) + mu * (z - y))

    def __iter__(self):
        self.iter: int = 0
        self.x = self.problem.x0.copy()
        self.y = self.x.copy()
        self.D = 0

        return super().__iter__()

    def __next__(self):
        self.calcLam()
        self.y = self.problem.Project(self.x - self.lam * self.problem.GradF(self.x))

        self.D = linalg.norm(self.x - self.y)

        if self.D >= self.eps or self.iter == 0 or self.min_iters > self.iter:
            self.iter += 1
            z = self._supportHProject(self.x - self.lam * self.problem.GradF(self.y))

            self.x = self.projectLast(self.xstar, self.x, (z + self.x) * 0.5)

            return self.currentState()
        else:
            raise StopIteration()

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, self.y), F=(self.problem.F(self.x), self.problem.F(self.y)),
                    D=self.D, lam=self.lam)

    def paramsInfoString(self) -> str:
        return super().paramsInfoString() + "; x0: {0};".format(self.problem.XToString(self.problem.x0))

    def currentStateString(self) -> str:
        return "{0}: j: {1}; lam: {2}; D: {3}; x: {4}; y: {5}; F(x): {6};".format(
            self.iter, self.j, scalarToString(self.lam), self.D,
            self.problem.XToString(self.x), self.problem.XToString(self.y),
            self.problem.FValToString(self.problem.F(self.x)))
