import os

import numpy as np

from utils.graph.alg_stat_grapher import XAxisType, YAxisType


class AlgorithmParams:
    def __init__(self, *,
                 eps: float = None,
                 x0: np.ndarray = None,
                 x1: np.ndarray = None,
                 lam: float = None,
                 start_adaptive_lam: float = None,
                 start_adaptive_lam1: float = None,
                 adaptive_tau: float = None,
                 adaptive_tau_large: float = None,
                 min_iters: int = None,
                 max_iters: int = None,
                 x_axis_type: XAxisType = XAxisType.ITERATION,
                 y_axis_type: YAxisType = YAxisType.STEP_DELTA
                 ):
        self.eps = eps

        self.x0 = x0
        self.x1 = x1

        self.lam = lam
        self.start_adaptive_lam = start_adaptive_lam
        self.start_adaptive_lam1 = start_adaptive_lam1
        self.adaptive_tau = adaptive_tau
        self.adaptive_tau_large = adaptive_tau_large

        self.max_iters = max_iters
        self.min_iters = min_iters

        self.y_axis_type = y_axis_type
        self.x_axis_type = x_axis_type

    def saveToDir(self, path: str):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([
                f"eps:{self.eps}\n",
                f"lam:{self.lam}\n",
                f"start_adaptive_lam:{self.start_adaptive_lam}\n",
                f"start_adaptive_lam1:{self.start_adaptive_lam1}\n",
                f"adaptive_tau:{self.adaptive_tau}\n",
                f"adaptive_tau_large:{self.adaptive_tau_large}\n",
                f"max_iters:{self.max_iters}\n",
                f"min_iters:{self.min_iters}\n"
                f"x0:{np.array2string(self.x0, max_line_width=100000)}\n",
                f"x1:{np.array2string(self.x1, max_line_width=100000)}\n",
                f"x_axis_type:{self.x_axis_type}\n",
                f"y_axis_type:{self.y_axis_type}\n",
            ])
