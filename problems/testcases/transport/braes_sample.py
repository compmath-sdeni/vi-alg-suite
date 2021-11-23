import numpy as np

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams
from problems.traffic_equilibrium import TrafficEquilibrium
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def G(x: np.ndarray) -> np.ndarray:
    return np.array([
        10.*x[0] + 10.*x[2] + x[0] + 50,
        x[1] + 50 + 10.*x[1] + 10.*x[2],
        10.*x[0] + 10.*x[2] + x[2] + 10 + 10.*x[1] + 10.*x[2]
    ])


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    # region Test problem one - 3x3
    n = 3
    W = np.array([
        [0, 1, 2]
    ])

    d = np.array([6.])

    real_solution = np.array([2., 2., 2.])

    algorithm_params.x0 = np.array([3., 2., 1.])
    algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion

    algorithm_params.eps = 1e-5
    algorithm_params.max_iters = 300

    algorithm_params.lam = 0.01
    algorithm_params.lam_medium = 0.1
    algorithm_params.lam_small = 0.1

    algorithm_params.min_iters = 3

    algorithm_params.start_adaptive_lam = 1.0
    algorithm_params.start_adaptive_lam1 = 1.0

    algorithm_params.adaptive_tau = 0.9
    algorithm_params.adaptive_tau_small = 0.45

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    algorithm_params.y_label = "Gap"
    # algorithm_params.x_label = "sec."

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 3

    return TrafficEquilibrium(
        Gf=G, d=d, W=W, C=Rn(n),
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ traffic equilibrium ' +
                #                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                #                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_small, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
