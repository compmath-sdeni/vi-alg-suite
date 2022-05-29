import numpy as np

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams
from problems.traffic_equilibrium import TrafficEquilibrium
from utils.graph.alg_stat_grapher import YAxisType, XAxisType
from problems.testcases.transport.transportation_network import TransportationNetwork, EdgeParams
from matplotlib import pyplot as plt


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    tnet = TransportationNetwork(
        edges_list=[
            (1, 3, {str(EdgeParams.FRF): 0.00000001, str(EdgeParams.K): 1000000000., str(EdgeParams.CAP): 1., str(EdgeParams.POW): 1.}),
            (1, 4, {str(EdgeParams.FRF): 50., str(EdgeParams.K): 0.02, str(EdgeParams.CAP): 1., str(EdgeParams.POW): 1.}),
            (3, 2, {str(EdgeParams.FRF): 50., str(EdgeParams.K): 0.02, str(EdgeParams.CAP): 1., str(EdgeParams.POW): 1.}),
            (3, 4, {str(EdgeParams.FRF): 10., str(EdgeParams.K): 0.1, str(EdgeParams.CAP): 1., str(EdgeParams.POW): 1.}),
            (4, 2, {str(EdgeParams.FRF): 0.00000001, str(EdgeParams.K): 1000000000., str(EdgeParams.CAP): 1., str(EdgeParams.POW): 1.}),
        ],
        demand=[(1, 2, 6.)]
    )

    tnet.show()

    tnet.draw()
    plt.show()

    d = tnet.get_demands_vector()

    Q: np.ndarray = tnet.Q
    Gf = tnet.get_cost_function()
    W = tnet.get_paths_to_demands_incidence()
    n = Q.shape[1]

    if n < 10:
        print("Demands: ")
        print(d)

        print("Edges to paths incidence: ")
        print(Q)

        print("Demands to paths incidence: ")
        print(W)
    else:
        print(f"Demands count: {len(d)}")
        print(f"Paths count: {n}")

    #real_solution = np.array([2., 2., 2.])
    real_solution = None

    if real_solution:
        print(f"Cost from real solution (from {real_solution[:5]})")
        print(Gf(real_solution)[:5])

    algorithm_params.x0 = np.array([3., 3., 0.])

    #algorithm_params.x0 = np.array([0.1 for i in range(n)])

    print(f"Cost from initial flow (from {algorithm_params.x0[:5]} ...)")
    print(Gf(algorithm_params.x0)[:5])

    print(f"Initial flow on edges:")
    print((tnet.Q @ algorithm_params.x0)[:10])

    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.eps = 1e-8
    algorithm_params.max_iters = 1000

    algorithm_params.lam = 0.05
    algorithm_params.lam_medium = 0.0000001
    algorithm_params.lam_KL = 0.1

    algorithm_params.min_iters = 3

    algorithm_params.start_adaptive_lam = 0.1
    algorithm_params.start_adaptive_lam1 = 0.1

    algorithm_params.adaptive_tau = 0.5
    algorithm_params.adaptive_tau_small = 0.25

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    algorithm_params.y_label = "Gap"
    # algorithm_params.x_label = "sec."

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 3

    problem = TrafficEquilibrium(
        Gf=Gf, d=d, W=W, C=Rn(n), Q=Q,
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ traffic equilibrium ' +
                #                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                #                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )

    print(f"Initial data: Max individual loss = {problem.getIndividualLoss(problem.x0)}; Gap = {problem.F(problem.x0)}")

    return problem
