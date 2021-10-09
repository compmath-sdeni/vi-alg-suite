import numpy as np
import nashpy as nash

from constraints.allspace import Rn
from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.minmax_game import MinMaxGame
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    # Random problem
    n = 1000
    m = 1000
    #P = np.random.randint(-3, 4, size=(m,n)).astype(float)
    # P = np.random.rand(m, n)
    # P = np.random.normal(0., 10., (m, n))
    real_solution = None

    #np.save('minmax_P_1000_symmetric_-3_3', P)
    P = np.load('minmax_P_1000_symmetric_-3_3.npy')

    algorithm_params.x0 = np.concatenate((np.array([1. / m for i in range(m)]), np.array([1. / n for i in range(n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()

    # Test problem one
    # n = 3
    # m = 3
    # P = np.array([[1,-1,-1], [-1,-1,3],[-1,3,-1]])
    # real_solution = np.array([0.5, 0.25, 0.25, 0.5, 0.25, 0.25])
    #
    # algorithm_params.x0 = np.concatenate((np.array([1./m for i in range(m)]), np.array([1./n for i in range(n)])))
    # algorithm_params.x1 = algorithm_params.x0.copy()

    # Test Blotto game
    # n = 5
    # m = 4
    # P = np.array([
    #     [4, 2, 1, 0],
    #     [1, 3, 0, -1],
    #     [-2, 2, 2, -2],
    #     [-1, 0, 3, 1],
    #     [0, 1, 2, 4],
    # ])
    # real_solution = None

    # rps = nash.Game(P)
    # print(rps)
    # eqs = list(rps.support_enumeration())
    # real_solution = np.concatenate((np.array(eqs[0]), np.array(eqs[1])))
    # print(real_solution)

    algorithm_params.eps = 1e-5
    algorithm_params.max_iters = 1500

    algorithm_params.lam = 0.9 / np.linalg.norm(P, 2)
    algorithm_params.lam_small = 0.45 / np.linalg.norm(P, 2)

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    algorithm_params.adaptive_tau = 0.95
    algorithm_params.adaptive_tau_small = 0.45

    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.x_axis_type = XAxisType.TIME
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    algorithm_params.y_label = "duality gap"

    algorithm_params.time_scale_divider = 1e+9
    algorithm_params.x_label = "Time, sec."

    algorithm_params.styles = [['b-'], ['b--'], ['g--'], ['g-']]
    algorithm_params.plot_start_iter = 2

    return MinMaxGame(
        P=P, C=Rn(n + m),
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ min max (Px,y) ' +
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_small, 5)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
