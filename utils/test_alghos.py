import time
from typing import Callable

import numpy as np

from methods.IterativeAlgorithm import IterativeAlgorithm


class BasicAlghoTests:
    def __init__(self, *, min_time: float = 0, print_every: int = 1, max_iters: int = 1000,
                 on_alg_start: Callable[[IterativeAlgorithm, dict], None] = None, on_alg_finish: Callable[[IterativeAlgorithm, dict], None] = None,
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
                self.on_alg_start(alg, alg.currentState()) # dict()

            start: float = time.perf_counter()
            prev_time: float = 0
            # noinspection PyBroadException
            if self.min_time > 0:
                prev_time = time.perf_counter()

            for curState in alg:
                if self.on_iteration is not None:
                    cur: float = time.perf_counter()
                    if not self.on_iteration(alg, curState, iter_num, cur - start):
                        return True

                if (iter_num == 1 and self.print_every > 0) or (
                                self.print_every > 0 and iter_num % self.print_every == 0):
                    print(alg.currentStateString())
                    # print("{0}: {1} -> {2}; err: {3}; {4}".format(iter, alg.problem.XToString(curState['x']),
                    # alg.problem.FValToString(v), scalartostring(alg.problem.GetErrorByTestX(curState['x'])),
                    # extra if extra is not None else ''))

                iter_num += 1
                if self.print_every >= 0 and iter_num > self.max_iters:
                    print("Maximum iterations number reached!")
                    break

                if self.min_time > 0:
                    tm = time.perf_counter()
                    if (tm - prev_time) < self.min_time:
                        time.sleep(self.min_time - (tm - prev_time))

                    prev_time = time.perf_counter()

            end:float = time.perf_counter()
            stats[alg_name]['iterations'] = iter_num
            stats[alg_name]['time'] = (end - start)
            stats[alg_name]['solution'] = curState['x'] if not isinstance(curState['x'], (list, tuple, np.ndarray)) \
                else curState['x'][0]

            if 'err' in curState:
                stats[alg_name]['err'] = curState['err']

            stats[alg_name]['test_error'] = alg.GetErrorByTestX(stats[alg_name]['solution'])

            print(alg_name + " done {0} iterations in {1} seconds. Result:\n{2}".format(
                iter_num, end - start, alg.currentStateString()))

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
