import numpy as np
import pandas as pd
from openpyxl import load_workbook


class AlgHistory:
    def __init__(self, dim:int, max_iters: int = 10000):
        self.alg = None

        self.iters_count = 0
        self.projections_count: int = 0
        self.operator_count: int = 0

        self.x = np.ndarray((max_iters, dim), dtype=float)
        self.y = np.ndarray((max_iters, dim), dtype=float)

        self.lam = np.ndarray(max_iters, dtype=float)

        self.step_delta_norm = np.ndarray(max_iters, dtype=float)
        self.goal_func_value = np.ndarray(max_iters, dtype=float)

        self.iter_time_ns = np.ndarray(max_iters, dtype=int)
        self.real_error = np.ndarray(max_iters, dtype=float)

    def toPandasDF(self, *, labels: dict = None):
        x_res = []
        for i in range(self.iters_count):
            x_res.append(np.array2string(self.x[i], precision=5, separator=',', suppress_small=True))

        df = pd.DataFrame({
            "iteration": range(self.iters_count),
            "x": x_res,
            "lam": self.lam[:self.iters_count],
            "step_delta": self.step_delta_norm[:self.iters_count],
            "goal_func": self.goal_func_value[:self.iters_count],
            "time_ns": self.iter_time_ns[:self.iters_count],
            "real_error": self.real_error[:self.iters_count]
        }, copy=True)
        return df

