import numpy as np
import nashpy as nash
from numpy import inf

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.minmax_game import MinMaxGame
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 3
    m = 3
    P = np.array([[1,-1,-1], [-1,-1,3],[-1,3,-1]])
    real_solution = np.array([0.5, 0.25, 0.25, 0.5, 0.25, 0.25])

    algorithm_params.x0 = np.concatenate((np.array([1./m for i in range(m)]), np.array([1./n for i in range(n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()

    # rps = nash.Game(-P.T)
    # print(rps)
    # eqs = list(rps.support_enumeration())
    # print(eqs)
    #    real_solution = np.concatenate((np.array(eqs[0]), np.array(eqs[1])))
    #    print(real_solution)

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 1
    algorithm_params.stop_by = StopCondition.STEP_SIZE

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-5
    algorithm_params.max_iters = 500
    algorithm_params.min_iters = 50

    algorithm_params.lam = 0.5 / np.linalg.norm(P, 2)
    algorithm_params.lam_medium = 0.0  # 0.45 / np.linalg.norm(P, 2)
    # for Bregman variants
    algorithm_params.lam_KL = 0.5 / np.max(np.abs(P))  #  5.9 / (max(abs(np.max(P)), abs(np.min(P))))

    # algorithm_params.x_limits = [-0.1, 10.]
    # algorithm_params.y_limits = [0.02, 0.5]

    algorithm_params.start_adaptive_lam = 0.5
    algorithm_params.start_adaptive_lam1 = algorithm_params.start_adaptive_lam

    algorithm_params.adaptive_tau = 0.5 * 0.5
    algorithm_params.adaptive_tau_small = 0.33 * 0.5

    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR
    # algorithm_params.y_label = "Gap"
    # algorithm_params.x_label = "sec."
    # algorithm_params.y_limits = [1e-3,10]

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 0

    return MinMaxGame(
        P=P, C=Rn(n + m),
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ min \ max (Px,y) - small test.' +
#                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
#                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
