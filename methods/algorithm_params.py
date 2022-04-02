import os
from typing import List
from enum import Enum, unique

import numpy as np

from utils.graph.alg_stat_grapher import XAxisType, YAxisType

@unique
class StopCondition(Enum):
    STEP_SIZE = 0,
    GAP = 1,
    EXACT_SOL_DIST = 2


class AlgorithmParams:
    def __init__(self, *,
                 eps: float = None,
                 x0: np.ndarray = None,
                 x1: np.ndarray = None,
                 lam: float = None,
                 lam_spec: dict = None,
                 lam_KL: float = None,
                 lam_spec_KL: dict = None,
                 start_adaptive_lam: float = None,
                 start_adaptive_lam1: float = None,
                 adaptive_tau: float = None,
                 adaptive_tau_small: float = None,
                 min_iters: int = None,
                 max_iters: int = None,
                 x_axis_type: XAxisType = XAxisType.ITERATION,
                 y_axis_type: YAxisType = YAxisType.STEP_DELTA,
                 y_label: str = None,
                 x_label: str = None,
                 x_limits: list = None,
                 y_limits: list = None,
                 time_scale_divider: int = 1e+6,  # time is in nanoseconds - 1e+6 for ms, 1e+9 for sec.
                 styles: List[str] = None,
                 plot_start_iter: int = 2,
                 save_plots: bool = True,
                 show_plots: bool = True,
                 save_history: bool = True,
                 test_time: bool = False,
                 test_time_count: int = 10,
                 stop_by: StopCondition = StopCondition.STEP_SIZE
                 ):
        self.eps = eps

        self.x0 = x0
        self.x1 = x1

        self.lam = lam
        self.lam_spec = lam_spec
        self.lam_KL = lam_KL if lam_KL else lam
        self.lam_spec_KL = lam_spec_KL
        self.start_adaptive_lam = start_adaptive_lam
        self.start_adaptive_lam1 = start_adaptive_lam1
        self.adaptive_tau = adaptive_tau
        self.adaptive_tau_small = adaptive_tau_small

        self.max_iters = max_iters
        self.min_iters = min_iters

        self.y_axis_type = y_axis_type
        self.x_axis_type = x_axis_type
        self.y_label = y_label
        self.x_label = x_label
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.time_scale_divider = time_scale_divider
        self.styles = styles
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.plot_start_iter = plot_start_iter
        self.save_history = save_history

        self.stop_by = stop_by

        self.test_time = test_time
        self.test_time_count = test_time_count

    def saveToDir(self, path: str):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, self.__class__.__name__.lower() + ".txt"), "w") as file:
            file.writelines([
                f"eps:{self.eps}\n",
                f"lam:{self.lam}\n",
                f"lam_spec:{self.lam_spec}\n",
                f"lam_KL:{self.lam_KL}\n",
                f"lam_spec_KL:{self.lam_spec_KL}\n",
                f"start_adaptive_lam:{self.start_adaptive_lam}\n",
                f"start_adaptive_lam1:{self.start_adaptive_lam1}\n",
                f"adaptive_tau:{self.adaptive_tau}\n",
                f"adaptive_tau_small:{self.adaptive_tau_small}\n",
                f"max_iters:{self.max_iters}\n",
                f"min_iters:{self.min_iters}\n"
                f"x0:{np.array2string(self.x0, max_line_width=100000)}\n",
                f"x1:{np.array2string(self.x1, max_line_width=100000)}\n",
                f"x_axis_type:{self.x_axis_type}\n",
                f"y_axis_type:{self.y_axis_type}\n",
                f"x_label:{self.x_label}\n",
                f"y_label:{self.y_label}\n",
                f"time_scale_divider:{self.time_scale_divider}\n",
                f"styles:{self.styles}\n",
                f"plot_start_iter:{self.plot_start_iter}\n",
                f"save_plots:{self.save_plots}\n",
                f"show_plots:{self.show_plots}\n",
                f"save_history:{self.save_history}\n",
                f"stop_by:{self.stop_by}\n",
                f"test_time:{self.test_time}\n",
                f"test_time_count:{self.test_time_count}\n",
            ])
