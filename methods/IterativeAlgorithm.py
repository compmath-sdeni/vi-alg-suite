from problems.problem import Problem


class IterativeAlgorithm:
    def __init__(self, problem: Problem, eps: float = 0.0001, lam: float = 0.1, min_iters: int = 0):
        self.iter: int = 0
        self.problem: Problem = problem
        self.eps: float = eps
        self.lam: float = lam
        self.min_iters: int = min_iters

    def isStopConditionMet(self) -> float:
        return self.iter > self.min_iters

    def doStep(self):
        pass

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self) -> dict:
        if not self.isStopConditionMet():
            self.doStep()

            self.iter += 1
            return self.currentState()
        else:
            raise StopIteration()

    # noinspection PyMethodMayBeStatic
    def currentError(self) -> float:
        return 0

    def currentState(self) -> dict:
        return dict(iter=self.iter)

    def paramsInfoString(self) -> str:
        return "Eps: {0}; Lam: {1}".format(self.eps, self.lam)

    def currentStateString(self) -> str:
        return ''

    def GetErrorByTestX(self, x) -> float:
        return self.problem.GetErrorByTestX(x)
