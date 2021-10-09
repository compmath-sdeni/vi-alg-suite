from collections import deque
from methods.IterativeAlgorithm import IterativeAlgorithm
from problems.simplex_proj import SimplexProjProblem
from utils.print_utils import *


class SimplexProj(IterativeAlgorithm):
    def __init__(self, *, problem: SimplexProjProblem, eps: float = 0.0001, lam: float = 0.1, maxiters=1000, b:float = 1.0):
        super().__init__(problem, eps, lam)
        self.maxiters = maxiters
        self.n = self.problem.n
        self.z = self.problem.z
        self.b = b

    def __iter__(self):
        lam = (self.b - np.sum(self.z)) / self.n

        self.x = self.z + lam
        self.px = np.copy(self.x)
        return super().__iter__()

    def __next__(self):
        if self.iter == 0:
            self.iter += 1
            return self.currentState()

        neg = 0
        pos = 0
        posvals = 0.0
        for v in self.x:
            if v < 0:
                neg += 1
            elif v > 0:
                pos += 1
                posvals += v

        if neg == 0:
            raise StopIteration()

        lam = (self.b - posvals) / pos
        self.x[self.x < 0] = 0
        self.x[self.x > 0] += lam

        self.px = np.copy(self.x)
        self.iter += 1
        return self.currentState()

    def do(self, z: np.ndarray):  # get the last element
        self.z = z
        self.n = z.shape[0]
        dq = deque(self, maxlen=1)
        return dq.pop()['x'][0]

    @classmethod
    def doInplace(cls, z: np.ndarray, b: float = 1.0):

        lam = (b - np.sum(z)) / z.shape[0]
        z += lam

        neg = 0
        pos = 0
        posvals = 0.0
        for v in z:
            if v < 0:
                neg += 1
            elif v > 0:
                pos += 1
                posvals += v

        while neg > 0:
            lam = (b - posvals) / pos
            z[z < 0] = 0
            z[z > 0] += lam

            neg = 0
            pos = 0
            posvals = 0.0
            for v in z:
                if v < 0:
                    neg += 1
                elif v > 0:
                    pos += 1
                    posvals += v

    @classmethod
    def doB1BallProjInplace(cls, z: np.ndarray, b: float = 1.0):
        t = np.ones_like(z, int)
        t[z < 0] = -1
        z *= t
        if np.sum(z) > b:
            cls.doInplace(z, b)
        z *= t

    def currentError(self) -> float:
        return 0

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, self.problem.z))

    def GetErrorByTestX(self, x_extended) -> float:
        return self.problem.GetErrorByTestX(x_extended[:-1])

    def paramsInfoString(self) -> str:
        return "".format("{0}; x0: {1}", super().paramsInfoString(), vectorToString(self.problem.z))

    def currentStateString(self) -> str:
        return "{0}: x: {1}".format(self.iter, vectorToString(self.x))
