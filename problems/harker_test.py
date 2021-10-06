import os

from problems.viproblem import VIProblem
import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from constraints.allspace import Rn
from utils.print_utils import vectorToString


# noinspection PyPep8Naming
class HarkerTest(VIProblem):
    def __init__(self, M: int, C: ConvexSetConstraints = None,
                 x0: np.ndarray = None, hr_name: str = None, lam_override: float = None,
                 xtest: np.ndarray = None):
        super().__init__(x0=x0 if x0 is not None else np.ones(M), C=C, hr_name=hr_name, lam_override=lam_override, xtest=xtest)

        self.B = np.round(np.random.rand(M, M) * 5 - 2.5, 1)
        self.S = np.round(np.random.rand(M, M) * 5 - 2.5, 1)
        for i in range(M):
            self.S[i, i] = 0

        for i in range(M):
            for j in range(M):
                self.S[i, j] = -self.S[j, i]

        self.DM = np.identity(M, float)
        for i in range(M):
            self.DM[i, i] = np.round(np.random.rand() * 5. + 1.0)

        self.AM = self.B @ self.B.T + self.S + self.DM

        # self.q = np.random.rand(M) * -500.0
        self.q = np.zeros(M)

        self.norm = np.linalg.norm(self.AM, 2)
        print("HpHard norm: ", self.norm)

        # r = np.linalg.eig(self.A)
        # print("A:\n", self.A)
        # print("Eig:\n", r)

    def f(self, x: np.ndarray) -> float:
        return np.dot(x, x)

    def df(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def F(self, x: np.ndarray) -> float:
        return self.f(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.df(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.AM, x) + self.q

    def Project(self, x: np.ndarray) -> np.ndarray:
        return self.C.project(x) if self.C is not None else x

    def XToString(self, x: np.ndarray):
        return vectorToString(x)

    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        np.savetxt("{0}/{1}".format(path_to_save, 'AM.txt'), self.AM, delimiter=',', newline="],\n[")
        np.savetxt("{0}/{1}".format(path_to_save, 'DM.txt'), self.DM, delimiter=',', newline="],\n[")
        np.savetxt("{0}/{1}".format(path_to_save, 'S.txt'), self.S, delimiter=',', newline="],\n[")
        np.savetxt("{0}/{1}".format(path_to_save, 'B.txt'), self.B, delimiter=',', newline="],\n[")

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest, delimiter=',')

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0, delimiter=',')

        return path_to_save