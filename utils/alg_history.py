import os
from enum import Enum, unique

import numpy as np
import pandas as pd
from openpyxl import load_workbook

@unique
class AlgHistFieldNames(Enum):
    ALG_NAME = 'alg_name',
    ALG_CLASS = 'alg_class',
    ITERS_COUNT = 'iterations',
    PROJECTIONS_COUNT = 'projections_count',
    OPERATOR_COUNT = 'operator_count',
    X = 'x',
    Y = 'y',
    LAM = 'lam',
    STEP_DELTA_NORM = 'step_delta_norm',
    GOAL_FUNC_VALUE = 'goal_func_value',
    ITER_TIME_NS = 'iter_time_ns',
    REAL_ERROR = 'real_error',
    EXTRA_INDICATORS = 'extra_indicators'

    def __str__(self):
        return str(self.value[0])


class AlgHistory:
    def __init__(self, dim: int, max_iters: int = 10000):
        self.alg_name = None
        self.alg_class = None

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
        self.extra_indicators = []

    def toPandasDF(self, *, labels: dict = None):
        x_res = []
        for i in range(self.iters_count+1):
            x_res.append(np.array2string(self.x[i], precision=5, separator=',', suppress_small=True))

        y_res = []
        if self.y is not None:
            for i in range(self.iters_count+1):
                y_res.append(np.array2string(self.y[i], precision=5, separator=',', suppress_small=True))

        frame_columns = {
            AlgHistFieldNames.ITERS_COUNT: range(self.iters_count+1),
            AlgHistFieldNames.X: x_res,
            AlgHistFieldNames.Y: y_res,
            AlgHistFieldNames.LAM: self.lam[:self.iters_count+1],
            AlgHistFieldNames.STEP_DELTA_NORM: self.step_delta_norm[:self.iters_count+1],
            AlgHistFieldNames.GOAL_FUNC_VALUE: self.goal_func_value[:self.iters_count+1],
            AlgHistFieldNames.ITER_TIME_NS: self.iter_time_ns[:self.iters_count+1],
            AlgHistFieldNames.REAL_ERROR: self.real_error[:self.iters_count+1],
        }

        if len(self.extra_indicators) > 0:
            frame_columns[AlgHistFieldNames.EXTRA_INDICATORS] = self.extra_indicators

        df = pd.DataFrame(frame_columns, copy=True)
        return df

    def saveToDir(self, path: str):
        dir = path  # os.path.join(path, 'full_history')
        os.makedirs(dir, exist_ok=True)

        f = open(os.path.join(dir, f"params.txt"), "w")
        f.write(
            f"{AlgHistFieldNames.ALG_CLASS}:{self.alg_class}\n"
            f"{AlgHistFieldNames.ALG_NAME}:{self.alg_name}\n"
            f"{AlgHistFieldNames.ITERS_COUNT}:{self.iters_count}\n"
            f"{AlgHistFieldNames.PROJECTIONS_COUNT}:{self.projections_count}\n"
            f"{AlgHistFieldNames.OPERATOR_COUNT}:{self.operator_count}\n"
        )
        f.close()

        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.X)), self.x)
        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.Y)), self.y)
        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.LAM)), self.lam)
        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.STEP_DELTA_NORM)), self.step_delta_norm)
        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.GOAL_FUNC_VALUE)), self.goal_func_value)
        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.ITER_TIME_NS)), self.iter_time_ns)
        np.save("{0}/{1}".format(dir, str(AlgHistFieldNames.REAL_ERROR)), self.real_error)

    @classmethod
    def fromPandasDF(self, df: pd.DataFrame):
        iters_list: np.ndarray = df['iteration'].to_numpy()
        dim = iters_list.shape[0]

        res = AlgHistory(dim)
        res.x = df['x'].to_numpy()
        res.lam = df['lam'].to_numpy(copy=True, dtype=float)
        res.step_delta_norm = df['step_delta'].to_numpy(copy=True, dtype=float)
        res.goal_func_value = df['goal_func'].to_numpy(copy=True, dtype=float)
        res.iter_time_ns = df['time_ns'].to_numpy(copy=True, dtype=float)
        res.real_error = df['real_error'].to_numpy(copy=True, dtype=float)

        return res
