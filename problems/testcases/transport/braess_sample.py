import numpy as np

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams
from problems.traffic_equilibrium import TrafficEquilibrium
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    # region Test problem one - 3x3
    n = 3
    W = np.array([
        [0, 1, 2]
    ])

    # edges to paths incidence matrix
    Q = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=float)


    # affine edges cost function - cost(y) = Ay+b
    A = np.array([
        [10, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=float)
    b = np.array([0, 50, 0, 50, 10])

    # Affine cost function - with edges <-> paths relation using incidence matrix
    # 1. get edges flow by paths flow;
    # 2. Calculate edges cost by affine cost function;
    # 3. Transform edges cost to paths cost
    Ge = lambda x: Q.T @ (A @ (Q @ x) + b)

    d = np.array([6.])

    real_solution = np.array([2., 2., 2.])

    algorithm_params.x0 = np.array([0., 0., 6.])
    algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion

    algorithm_params.eps = 1e-9
    algorithm_params.max_iters = 3000

    algorithm_params.lam = 0.01
    algorithm_params.lam_medium = 0.1
    algorithm_params.lam_KL = 0.1

    algorithm_params.min_iters = 3

    algorithm_params.start_adaptive_lam = 1.0
    algorithm_params.start_adaptive_lam1 = 1.0

    algorithm_params.adaptive_tau = 0.9
    algorithm_params.adaptive_tau_small = 0.45

    algorithm_params.moving_average_window = 1

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    # algorithm_params.y_label = "Gap"
    # algorithm_params.x_label = "sec."

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 3

    return TrafficEquilibrium(
        Gf=Ge, d=d, W=W, C=Rn(n),Q=Q,
        x0=algorithm_params.x0,
        x_test=real_solution,
        # auto_update_structure=True,
        # structure_update_freq=1,
        hr_name='$ traffic equilibrium ' +
                #                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                #                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
