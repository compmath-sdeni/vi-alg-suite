import time
from typing import Union

import numpy as np

from problems.viproblem import VIProblem
from utils.alg_history import AlgHistory


class IterativeAlgorithm:
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters: int = 0, max_iters: int = 5000, hr_name: str = None):
        self.iter: int = 0
        self.projections_count: int = 0
        self.operator_count: int = 0
        self.problem: VIProblem = problem
        self.eps: float = eps
        self.lam: float = lam
        self.iterEndTime = 0
        self.max_iters = max_iters
        self.zero_delta = 1e-20

        self.x: Union[np.ndarray, float] = self.problem.x0.copy()

        if isinstance(problem.x0, np.ndarray):
            self.N = problem.x0.shape[0]
        else:
            self.N = 1

        self.history = AlgHistory(self.N)

        if hasattr(self.problem, 'lam_override_by_method') and self.problem.lam_override_by_method is not None:
            if type(self).__name__ in self.problem.lam_override_by_method:
                self.lam = self.problem.lam_override_by_method[type(self).__name__]
        else:
            if hasattr(self.problem, 'lam_override') and self.problem.lam_override is not None:
                self.lam = self.problem.lam_override

        self.min_iters: int = min_iters
        self.hr_name = hr_name if hr_name else type(self).__name__

    def isStopConditionMet(self) -> float:
        return self.iter >= self.max_iters

    def doStep(self):
        pass

    def setHistoryData(self, *, x: np.ndarray = None, y: np.ndarray = None,
                       step_delta_norm: float = None, goal_func_value: float = None):
        if x is not None:
            self.history.x[self.iter] = x

        if y is not None:
            self.history.y[self.iter] = y

        if step_delta_norm is not None:
            self.history.step_delta_norm[self.iter] = step_delta_norm

            # we have no step error info on the step-0 - copy it from step 1
            if self.iter == 1:
                self.history.step_delta_norm[0] = step_delta_norm

        if goal_func_value is not None:
            self.history.goal_func_value[self.iter] = goal_func_value

    def doPostStep(self):
        pass

    def __iter__(self):
        self.iter = 0
        self.iterEndTime = 0
        self.projections_count: int = 0
        self.operator_count: int = 0

        self.history = AlgHistory(self.N)
        self.history.alg_name = self.hr_name
        self.history.alg_class = self.__class__.__name__

        return self

    def __next__(self) -> dict:
        if self.iter <= self.min_iters or (not self.isStopConditionMet()):

            start = time.process_time_ns()
            self.doStep()
            finish = time.process_time_ns()
            self.iterEndTime = time.process_time()

            self.history.projections_count = self.projections_count
            self.history.operator_count = self.operator_count

            self.history.iters_count = self.iter + 1
            self.history.iter_time_ns[self.iter] = (finish - start) + (
            self.history.iter_time_ns[self.iter - 1] if self.iter > 0 else 0)
            self.history.lam[self.iter] = self.lam

            if self.problem.xtest is not None:
                self.history.real_error[self.iter] = np.linalg.norm(self.problem.xtest - self.x[:self.problem.xtest.shape[0]])

            self.doPostStep()

            self.iter += 1
            # return self.currentState()
        else:
            raise StopIteration()

    def do(self):
        for curState in self:
            pass

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

    def GetHRName(self):
        return self.hr_name if self.hr_name is not None else self.__class__.__name__
