import time
from typing import Union, Dict

import numpy as np

from methods.algorithm_params import StopCondition, AlgorithmParams
from problems.viproblem import VIProblem
from utils.alg_history import AlgHistory


class IterativeAlgorithm:
    def __init__(self, problem: VIProblem, eps: float = 0.0001, lam: float = 0.1, *,
                 min_iters: int = 0, max_iters: int = 5000, hr_name: str = None,
                 stop_condition: StopCondition = StopCondition.STEP_SIZE, save_history: bool = True,
                 moving_average_window: int | None = None
                 ):
        self.iter: int = 0
        self.projections_count: int = 0
        self.operator_count: int = 0
        self.problem: VIProblem = problem
        self.eps: float = eps
        self.lam: float = lam
        self.iterEndTime = 0
        self.totalTime: int = 0
        self.max_iters = max_iters
        self.zero_delta = 1e-20
        self.stop_condition = stop_condition
        self.save_history = save_history

        self.x: Union[np.ndarray, float] = self.problem.x0.copy()

        if isinstance(problem.x0, np.ndarray):
            self.N = problem.x0.shape[0]
        else:
            self.N = 1

        self.moving_average_window = moving_average_window

        if self.moving_average_window is not None and self.moving_average_window > 0:
            self.averaging_buffer: list[Union[np.ndarray, float]] = []
        else:
            self.averaging_buffer = None

        self.averaged_result: Union[np.ndarray, float] | None = None

        if hasattr(self.problem, 'lam_override_by_method') and self.problem.lam_override_by_method is not None:
            if type(self).__name__ in self.problem.lam_override_by_method:
                self.lam = self.problem.lam_override_by_method[type(self).__name__]
        else:
            if hasattr(self.problem, 'lam_override') and self.problem.lam_override is not None:
                self.lam = self.problem.lam_override

        self.min_iters: int = min_iters
        self.hr_name = hr_name if hr_name else type(self).__name__

        self.history = AlgHistory(self.N, self.max_iters + 2 if self.save_history else 2)
        self.history.alg_name = self.hr_name
        self.history.alg_class = self.__class__.__name__

    def isStopConditionMet(self) -> bool:
        return self.iter >= self.max_iters

    def doStep(self):
        pass

    def setHistoryData(self, *, x: np.ndarray = None, y: np.ndarray = None,
                       step_delta_norm: float = None, goal_func_value: float = None,
                       goal_func_from_average: float = None):

        history_item_index: int = self.iter if self.save_history else (0 if self.iter == 0 else 1)

        if x is not None:
            self.history.x[history_item_index] = x

        if y is not None:
            self.history.y[history_item_index] = y

        if step_delta_norm is not None:
            self.history.step_delta_norm[history_item_index] = step_delta_norm

            # we have no step error info on the step-0 - copy it from step 1
            if self.iter == 1:
                self.history.step_delta_norm[0] = step_delta_norm

        if goal_func_value is not None:
            self.history.goal_func_value[history_item_index] = goal_func_value

        if goal_func_from_average is not None:
            self.history.goal_func_from_average[history_item_index] = goal_func_from_average

            # if self.iter == 3:
            #     self.history.goal_func_from_average[2] = goal_func_from_average
            #     self.history.goal_func_from_average[1] = goal_func_from_average
            #     self.history.goal_func_from_average[0] = goal_func_from_average

    def doPostStep(self):
        pass

    def __iter__(self):
        self.iter = 0
        self.iterEndTime = 0
        self.projections_count: int = 0
        self.operator_count: int = 0

        self.totalTime = 0

        # old averaging approach - full average
        # self.cum_y = np.zeros_like(self.y)
        # self.averaged_result = None

        # moving average. If a window is None, no calculations are done. If the window size is < 0, do the full average.
        if self.moving_average_window is not None:
            if self.averaged_result is None: # can be set in a descendant class (if differs from x0)
                self.averaged_result = self.x.copy()
                
            if self.moving_average_window > 0:
                self.averaging_buffer.append(self.averaged_result)

        self.history.iter_time_ns[0] = 0
        self.history.lam[0] = self.lam

        if not self.save_history:
            self.history.extra_indicators = [None, None]

        if self.problem.xtest is not None:
            self.history.real_error[0] = np.linalg.norm(self.problem.xtest - self.x[:self.problem.xtest.shape[0]])

        self.doPostStep()

        extra = self.problem.GetExtraIndicators(self.x, averaged_x=self.averaged_result)
        if extra:
            if self.save_history:
                self.history.extra_indicators.append(extra)
            else:
                self.history.extra_indicators[0] = extra

        return self

    def __next__(self) -> dict:

        if self.iter <= self.min_iters or (not self.isStopConditionMet()):
            self.iter += 1
            start = time.process_time_ns()
            self.doStep()

            if self.problem.zero_cutoff is not None:
                self.x[self.x < self.problem.zero_cutoff] = 0

            finish = time.process_time_ns()
            self.iterEndTime = time.process_time()

            history_index: int = self.iter if self.save_history else 1

            self.history.projections_count = self.projections_count
            self.history.operator_count = self.operator_count

            self.history.iters_count = self.iter + 1

            if self.save_history:
                self.totalTime = (finish - start) + (
                    self.history.iter_time_ns[history_index - 1] if self.iter > 0 else 0)
            else:
                self.totalTime = (finish - start) + (self.history.iter_time_ns[history_index] if self.iter > 0 else 0)

            self.history.iter_time_ns[history_index] = self.totalTime
            self.history.lam[history_index] = self.lam

            if self.problem.xtest is not None:
                self.history.real_error[history_index] = np.linalg.norm(
                    self.problem.xtest - self.x[:self.problem.xtest.shape[0]])

            extra = self.problem.GetExtraIndicators(self.x, averaged_x=self.averaged_result)
            if extra:
                if self.save_history:
                    self.history.extra_indicators.append(extra)
                else:
                    self.history.extra_indicators[history_index] = extra

            self.doPostStep()

            if self.problem.auto_update_structure and self.iter > 0 and self.iter % self.problem.structure_update_freq == 0:
                self.problem.updateStructure(self.x)

            # return self.currentState()
        else:
            raise StopIteration()

    def update_average_result(self, next_val: Union[np.ndarray, float]):
        if self.moving_average_window is not None:
            if self.moving_average_window < 0 or self.iter < self.moving_average_window:
                # self.cum_y += self.y
                # self.averaged_result = self.cum_y / self.iter
                self.averaged_result = self.averaged_result * (self.iter / (self.iter + 1)) + next_val / (self.iter + 1)
                self.averaging_buffer.append(next_val)
            else:
                self.averaged_result = self.averaged_result + (
                            next_val - self.averaging_buffer[self.iter % self.moving_average_window]) / self.moving_average_window
                self.averaging_buffer[self.iter % self.moving_average_window] = next_val


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

    def isAdaptive(self) -> bool:
        return False
