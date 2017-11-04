from scipy import linalg

from methods.IterativeAlgorithm import IterativeAlgorithm
from problems.convex_poly_proj import ConvexPolyProjProblem
from tools.print_utils import *


class PolytopeProj(IterativeAlgorithm):
    def __init__(self, *, problem: ConvexPolyProjProblem, eps: float = 0.0001, lam: float = 0.1, maxiters=1000):
        super().__init__(problem, eps, lam)
        self.maxiters = maxiters

    def __iter__(self):
        self.x = np.copy(self.problem.z)
        self.px = np.copy(self.x)
        self.t = np.copy(self.x)
        return super().__iter__()

    def __next__(self):
        if self.iter == 0:
            isIn, failedConstraint = self.problem.polytope.isIn(self.x)
            if isIn:
                self.iter += 1
                return self.currentState()

        if self.iter > 0 and linalg.norm(self.x - self.px) < self.eps:
            raise StopIteration()

        m = len(self.problem.polytope.constraints)
        j = i = self.iter % m

        while np.dot(self.problem.polytope.constraints[j % m][0], self.x) - self.eps < \
                self.problem.polytope.constraints[j % m][1]:
            j += 1
            if j - m >= i:
                self.px = np.copy(self.x)
                return self.currentState()

        constr = self.problem.polytope.constraints[j % m]

        self.px = np.copy(self.x)
        self.t = self.x + ((constr[1] - np.dot(constr[0], self.x)) * constr[0]) / np.dot(constr[0], constr[0])

        self.x = (1.0 / (self.iter + 2)) * self.problem.z + (1.0 - 1.0 / (self.iter + 2)) * self.t
        self.iter += 1
        return self.currentState()

        # print("t: {0}; iter:{1}; p: {2}; failed: {3}".format(t, self.iter, p, failedConstr))
        # pp = np.copy(p);
        # for i in range(n):
        #     constr = self.constraints[i]
        #     p[i] = t/lam + pp[i] - ((constr[1] - np.dot(constr[0], (t+lam*pp[i]))) * constr[0]) / (np.dot(constr[0], constr[0]) * lam)
        #
        # t = self.x - np.sum(p, axis=0)

        # t = t + ((failedConstr[1] - np.dot(failedConstr[0], t)) * failedConstr[0]) / np.dot(
        #     failedConstr[0], failedConstr[0])

    def currentError(self) -> float:
        return 0

    def currentState(self) -> dict:
        return dict(super().currentState(), x=(self.x, self.t, self.problem.z))

    def GetErrorByTestX(self, x_extended) -> float:
        return self.problem.GetErrorByTestX(x_extended[:-1])

    def paramsInfoString(self) -> str:
        return "".format("{0}; x0: {1}", super().paramsInfoString(), vectorToString(self.problem.z))

    def currentStateString(self) -> str:
        return "{0}: x: {1}; t: {2}".format(self.iter, vectorToString(self.x), vectorToString(self.t))
