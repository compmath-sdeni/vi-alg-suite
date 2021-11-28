import time
from typing import Callable

import numpy as np

from methods.IterativeAlgorithm import IterativeAlgorithm


class BasicAlgoTests:
    def __init__(self, *, min_time: float = 0, print_every: int = 1, max_iters: int = 1000,
                 on_alg_start: Callable[[IterativeAlgorithm, dict], None] = None,
                 on_alg_finish: Callable[[IterativeAlgorithm, dict], None] = None,
                 on_iteration: Callable[[IterativeAlgorithm, dict, int, float], None] = None,
                 on_all_finished: Callable[..., None] = None):
        self.print_every: int = print_every
        self.max_iters: int = max_iters
        self.on_iteration: Callable[[IterativeAlgorithm, dict, int, float], None] = on_iteration
        self.on_alg_start = on_alg_start
        self.on_alg_finish = on_alg_finish
        self.on_all_finished = on_all_finished
        self.min_time: float = min_time

    def DoTests(self, tested_items: list):
        stats: dict = {}
        for alg in tested_items:
            iter_num: int = 0
            alg_name: str = alg.hr_name if alg.hr_name is not None else type(alg).__name__
            probl_name: str = type(alg.problem).__name__
            print("\nStarting {0}. Params: {1}\nProblem: {2}".format(alg_name, alg.paramsInfoString(), probl_name))
            stats[alg_name] = {}
            stats[alg_name]['initial'] = alg.paramsInfoString()
            stats[alg_name]['problem'] = probl_name

            if self.on_alg_start is not None:
                self.on_alg_start(alg, alg.currentState())  # dict()

            prev_time: float = 0  # used only to ensure min iteration time if needed
            # noinspection PyBroadException
            if self.min_time > 0:
                prev_time = time.process_time()

            totalAlghoTime: float = 0
            referenceTime: float = time.process_time()  # used to count pure algho time, without output etc

            for curState in alg:
                totalAlghoTime += (curState['iterEndTime'] - referenceTime)

                if self.on_iteration is not None:
                    if not self.on_iteration(alg, curState, iter_num, totalAlghoTime):
                        return True

                if (iter_num == 1 and self.print_every > 0) or (
                        self.print_every > 0 and iter_num % self.print_every == 0):
                    print("{0:.3f}s. {1}".format(totalAlghoTime, alg.currentStateString()))
                    # print("{0}: {1} -> {2}; err: {3}; {4}".format(iter, alg.problem.XToString(curState['x']),
                    # alg.problem.FValToString(v), scalartostring(alg.problem.GetErrorByTestX(curState['x'])),
                    # extra if extra is not None else ''))

                iter_num += 1
                if self.print_every >= 0 and iter_num > self.max_iters:
                    print("Maximum iterations number reached!")
                    break

                if self.min_time > 0:
                    tm = time.process_time()
                    if (tm - prev_time) < self.min_time:
                        time.sleep(self.min_time - (tm - prev_time))

                    prev_time = time.process_time()

                referenceTime = time.process_time()

            stats[alg_name]['iterations'] = iter_num
            stats[alg_name]['time'] = totalAlghoTime
            stats[alg_name]['solution'] = curState['x'] if not isinstance(curState['x'], (list, tuple, np.ndarray)) \
                else curState['x'][0]

            if 'err' in curState:
                stats[alg_name]['err'] = curState['err']

            stats[alg_name]['test_error'] = alg.GetErrorByTestX(stats[alg_name]['solution'])

            print(alg_name + " done {0} iterations in {1} seconds. Result:\n{2}".format(
                iter_num, totalAlghoTime, alg.currentStateString()))

            if not (self.on_alg_finish is None):
                self.on_alg_finish(alg, curState)

                # except Exception:
                #     end = time.process_time()
                #     print("\n", type(alg).__name__,
                #           " done {0} iterations in {1} seconds, but exception occured!\n{0}".format(
                # iter_num, end - start, traceback.print_exc()),
                #           "\n")
                #     pass

        if self.on_all_finished is not None:
            self.on_all_finished()
        else:
            print('All tests finished.')

        return True

    @staticmethod
    def PrintAlgRunStats(alg_object: IterativeAlgorithm):
        print(f"{alg_object.hr_name} finished. "
              f"Iters: {alg_object.history.iters_count}; Projections: {alg_object.history.projections_count}; Operators calc: {alg_object.history.operator_count}; Time: {alg_object.history.iter_time_ns[alg_object.history.iters_count - 1] / 1e+9} sec.; "
              f"\nLast step: {alg_object.history.step_delta_norm[alg_object.history.iters_count - 1]}; "
              f"Exact error: {alg_object.history.real_error[alg_object.history.iters_count - 1]}; "
              f"Goal function: {alg_object.history.goal_func_value[alg_object.history.iters_count - 1]}; "
              f"Goal function form avg: {alg_object.history.goal_func_from_average[alg_object.history.iters_count - 1]}; "
              f"Lambda: {alg_object.history.lam[alg_object.history.iters_count - 1]}; "
              )

        if alg_object.problem.C:
            print(f"Distance to C: {alg_object.problem.C.getDistance(alg_object.x[:alg_object.problem.x_dim])}")

        print_size = 5

        cum_res:np.ndarray = None
        try:
            if alg_object.cum_x  is not None:
                cum_res = alg_object.cum_x/alg_object.iter
        except:
            try:
                if alg_object.cum_y is not None:
                    cum_res = alg_object.cum_y / alg_object.iter
            except:
                pass

        round_decimals = 5
        if alg_object.x.shape[0] <= print_size * 2:
            print("Result: {0}".format(alg_object.x))
            if cum_res is not None:
                print("Result AVG: {0}".format(cum_res))
        else:
            print_len = int(alg_object.x.shape[0] / 2)
            if print_len > print_size:
                print_len = print_size

            print("Result: {0} ... {1}\n".format(np.round(alg_object.x[:print_len], round_decimals), np.round(alg_object.x[-print_len:], round_decimals)))
            if cum_res is not None:
                print("Result AVG: {0} ... {1}\n".format(np.round(cum_res[:print_len], round_decimals), np.round(cum_res[-print_len:], round_decimals)))

        extra_indicators = alg_object.problem.GetExtraIndicators(alg_object.x)
        if extra_indicators:
            extra_strings = []
            for name, value in extra_indicators.items():
                extra_strings.append(f'{name}: {value}')

            print('; '.join(extra_strings))
