import numpy as np


class AlgHistory:
    def __init__(self, dim:int, max_iters: int = 10000):
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


