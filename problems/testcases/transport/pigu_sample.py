import numpy as np

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams
from problems.traffic_equilibrium import TrafficEquilibrium
from utils.graph.alg_stat_grapher import YAxisType, XAxisType

def G(x: np.ndarray) -> np.ndarray:
    # "original" variant - sol. 25, 5
    return np.array([50., 45. + x[1]])

    # "improved" variant - sol. 10, 20
    # return np.array([50., 40. + x[1] * 0.5])

def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 2
    W = np.array([
        [0, 1]
    ])

    # edges to paths incidence matrix
    Q = np.array([
        [1, 0],
        [0, 1],
    ], dtype=float)

    d = np.array([30.])

    real_solution = np.array([25., 5.])

    algorithm_params.x0 = np.array([15., 15.])
    algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion

    algorithm_params.eps = 1e-10
    algorithm_params.max_iters = 2000

    algorithm_params.lam = 0.1
    algorithm_params.lam_medium = 0.1
    algorithm_params.lam_KL = 0.1

    algorithm_params.min_iters = 3

    algorithm_params.start_adaptive_lam = 1.0
    algorithm_params.start_adaptive_lam1 = 1.0

    algorithm_params.adaptive_tau = 0.9
    algorithm_params.adaptive_tau_small = 0.45

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR
    algorithm_params.y_label = "$D_n$"
    # algorithm_params.x_label = "sec."

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 3

    return TrafficEquilibrium(
        Gf=G, d=d, W=W, C=Rn(n), Q=Q,
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ traffic - Pigu \ sample ' +
#                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
#                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
