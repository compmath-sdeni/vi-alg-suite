import os.path
import pathlib

import numpy as np

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams
from problems.traffic_equilibrium import TrafficEquilibrium
from utils.graph.alg_stat_grapher import YAxisType, XAxisType
from problems.testcases.transport.transportation_network import TransportationNetwork, EdgeParams
from matplotlib import pyplot as plt


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams(),
                   data_path: str,
                   net_file_name: str = 'sample_net.tntp',
                   demands_file_name: str = 'sample_trips.tntp',
                   pos_file_name: str = None, zero_cutoff: float = 0.5):

    tnet = TransportationNetwork()
    # tnet.load_network_graph(
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/data/TransportationNetworks/Braess-Example/Braess_net.tntp',
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/data/TransportationNetworks/Braess-Example/Braess_trips.tntp')

    # tnet.load_network_graph(
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/data/TransportationNetworks/SiouxFalls/SiouxFalls_net.tntp',
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/data/TransportationNetworks/SiouxFalls/SiouxFalls_trips.tntp')

    tnet.load_network_graph(
        os.path.join(data_path, net_file_name),
        os.path.join(data_path, demands_file_name),
        max_od_paths_count=10,
        max_path_edges=25,
        saved_paths_file=os.path.join(data_path, 'saved_paths_cnt10_depth25.npy'),
        pos_file=os.path.join(data_path, pos_file_name) if pos_file_name is not None else None)

    tnet.show(limit=50)

    # tnet.draw()
    # plt.show()

    d = tnet.get_demands_vector()

    Q: np.ndarray = tnet.Q
    Gf = tnet.get_cost_function()
    W = tnet.get_paths_to_demands_incidence()
    n = Q.shape[1]

    if n <= 10:
        print("Demands: ")
        print(d)

        print("Edges to paths incidence: ")
        print(Q)

        print("Demands to paths incidence: ")
        print(W)
    else:
        print(f"Demands count: {len(d)}")
        print(f"Paths count: {n}")

    real_solution = None
    # real_solution = np.array([5., 4., 3., 3.])

    if real_solution is not None:
        print(f"Cost from real solution (from {real_solution[:5]})")
        print(Gf(real_solution)[:10])

        print(f"Flow on edges from real solution:")
        print(Q.dot(real_solution)[:10])

    #algorithm_params.x0 = np.array([6., 0., 0.])

    algorithm_params.x0 = np.array([0. for i in range(n)])
    # build initial flow
    for idx, demand in enumerate(tnet.get_demands_vector()):
        paths_for_pair = W[idx]
        flow = demand / len(paths_for_pair)
        for path_index in paths_for_pair:
            algorithm_params.x0[path_index] = flow

    print(f"Cost from initial flow (from {algorithm_params.x0[:10]} ...)")
    print(Gf(algorithm_params.x0)[:10])

    print(f"Initial flow on edges:")
    print((tnet.Q @ algorithm_params.x0)[:10])

    algorithm_params.save_history = False

    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.eps = 1e-8
    algorithm_params.max_iters = 500

    algorithm_params.lam = 0.1
    algorithm_params.lam_medium = 0.00001
    algorithm_params.lam_KL = 0.1

    algorithm_params.min_iters = 3

    algorithm_params.start_adaptive_lam = 0.0005
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

    problem = TrafficEquilibrium(
        Gf=Gf, d=d, W=W, Q=Q, C=Rn(n),
        zero_cutoff=zero_cutoff,
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ traffic equilibrium ' +
                #                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                #                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )

    print(f"Initial state:\n{problem.GetExtraIndicators(problem.x0)}")

    return problem
