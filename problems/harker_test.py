import os

from problems.viproblem import VIProblem
import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints
from constraints.allspace import Rn
from utils.print_utils import vectorToString


# noinspection PyPep8Naming
class HarkerTest(VIProblem):
    def __init__(self, M: int, *, C: ConvexSetConstraints = None,
                 matr: np.ndarray = None, q: np.ndarray = None,
                 x0: np.ndarray = None, hr_name: str = None, unique_name: str = 'HarkerTestProblem',
                 lam_override: float = None, xtest: np.ndarray = None):

        super().__init__(x0=x0 if x0 is not None else np.ones(M), C=C, hr_name=hr_name, unique_name=unique_name,
                         lam_override=lam_override, xtest=xtest)

        if matr is None:
            self.M = M
            self.B = np.round(np.random.rand(M, M) * 5 - 2.5, 1)
            self.S = np.round(np.random.rand(M, M) * 5 - 2.5, 1)
            for i in range(M):
                self.S[i, i] = 0

            for i in range(M):
                for j in range(M):
                    self.S[i, j] = -self.S[j, i]

            self.DM = np.identity(M, float)
            for i in range(M):
                self.DM[i, i] = np.round(np.random.rand() * 0.3)

            self.AM = self.B @ self.B.T + self.S + self.DM
        else:
            self.AM = matr
            self.M = self.AM.shape[0]
            self.DM = None
            self.S = None
            self.D = None
            self.B = None

        if q is None:
            self.q = np.random.rand(self.M) * 5.0 - 5.
        else:
            self.q = q

        #self.q = np.zeros(M)

        self.norm = np.linalg.norm(self.AM, 2)
        print("HpHard norm: ", self.norm)

        # r = np.linalg.eig(self.A)
        # print("A:\n", self.A)
        # print("Eig:\n", r)

    def setParams(self, matr: np.ndarray, q:np.ndarray):
        self.AM = matr
        self.q = q
        self.M = q.shape[0]
        self.norm = np.linalg.norm(self.AM, 2)

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

        if self.DM is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'DM.txt'), self.DM, delimiter=',', newline="],\n[")
        if self.S is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'S.txt'), self.S, delimiter=',', newline="],\n[")
        if self.B is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'B.txt'), self.B, delimiter=',', newline="],\n[")

        np.savetxt("{0}/{1}".format(path_to_save, 'q.txt'), self.q, delimiter=',', newline="],\n[")

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest, delimiter=',')

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0, delimiter=',')

        return path_to_save