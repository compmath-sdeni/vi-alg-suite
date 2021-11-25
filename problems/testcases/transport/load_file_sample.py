import numpy as np

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams
from problems.traffic_equilibrium import TrafficEquilibrium
from utils.graph.alg_stat_grapher import YAxisType, XAxisType
from problems.testcases.transport.transportation_network import TransportationNetwork
from matplotlib import pyplot as plt


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    tnet = TransportationNetwork(
        edges_list=[
            (1, 3, {'frf': 0.00000001, 'k': 1000000000., 'cap': 1., 'pow': 1.}),
            (1, 4, {'frf': 50., 'k': 0.02, 'cap': 1., 'pow': 1.}),
            (3, 2, {'frf': 50., 'k': 0.02, 'cap': 1., 'pow': 1.}),
            (3, 4, {'frf': 10., 'k': 0.1, 'cap': 1., 'pow': 1.}),
            (4, 2, {'frf': 0.00000001, 'k': 1000000000., 'cap': 1., 'pow': 1.}),
        ],
        demand=[(1, 2, 6.)]
    )

    tnet.calc_paths()
    tnet.show()

    # tnet.draw()
    # plt.show()

    d = tnet.get_demands_vector()
    print("Demands: ")
    print(d)

    Q: np.ndarray = tnet.Q
    print("Edges to paths incidence: ")
    print(Q)

    Gf = tnet.get_cost_function()

    W = tnet.get_paths_to_demands_incidence()
    print("Path to demands:")
    print(W)

    real_solution = np.array([2., 2., 2.])

    print(f"Cost from real solution (from {real_solution})")
    print(Gf(real_solution))

    algorithm_params.x0 = np.array([6., 0., 0.])
    algorithm_params.x1 = algorithm_params.x0.copy()


    # region Braess example - hardcoded
    # n = 3

    # paths to demands incidence matrix
    # Have a row for each demand (source-destination pair)
    # And the row contains vector of path indices which should satisfy corresponding demand
    # W = [
    #     np.array([0, 1, 2])
    # ]

    # edges to paths incidence matrix
    # Q = np.array([
    #     [1, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1],
    #     [1, 0, 0],
    #     [0, 0, 1]
    # ], dtype=float)

    # affine edges cost function - cost(y) = Ay+b
    # A = np.array([
    #     [10, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0],
    #     [0, 0, 10, 0, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1],
    # ], dtype=float)
    # b = np.array([0, 50, 0, 50, 10])

    # Affine cost function - with edges <-> paths relation using incidence matrix
    # 1. get edges flow by paths flow;
    # 2. Calculate edges cost by affine cost function;
    # 3. Transform edges cost to paths cost
    # Ge = lambda x: Q.T @ (A @ (Q @ x) + b)

    # d = np.array([6.])

    # real_solution = np.array([2., 2., 2.])
    #
    # algorithm_params.x0 = np.array([0., 0., 6.])
    # algorithm_params.x1 = algorithm_params.x0.copy()
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
        Gf=Gf, d=d, W=W, C=Rn(len(W)),
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ traffic equilibrium ' +
                #                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                #                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_small, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
