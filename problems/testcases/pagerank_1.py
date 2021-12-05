import numpy as np
from numpy import inf
import networkx as nx
import pandas as pd

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.page_rank_problem import PageRankProblem
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    node_labels: list = None

    # region NetworkX based random graph

    # n = 4
    # G: nx.DiGraph = nx.DiGraph([
    #     (1, 2), (1, 4),
    #     (2, 3),
    #     (3, 4),
    #     (4, 1)
    # ])

    n = 1000

    # G: nx.DiGraph = nx.generators.scale_free_graph(n)
    # nx.readwrite.write_edgelist(G, f'pagerank_G_{n}.edges')

    G: nx.DiGraph = nx.readwrite.read_edgelist(f'pagerank_G_{n}.edges', create_using=nx.DiGraph)
    GraphMatr:np.ndarray = np.array(nx.google_matrix(G)).T

    real_solution: np.ndarray = np.zeros(n, dtype=float)
    pr_dict = nx.pagerank_numpy(G)
    for i, k in enumerate(pr_dict):
        real_solution[i] = pr_dict[k]

    # # print(f"Test PR: {real_solution}")
    print(f"Test on real solution: {np.max(np.abs(GraphMatr @ real_solution - real_solution))}")
    # endregion

    # region California search edges list
    # https://www.quora.com/Where-can-I-find-arff-data-sets-for-implementing-page-rank
    # https://www.cs.cornell.edu/courses/cs685/2002fa/data/gr0.California

    # storage_dir: str = '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/data/PageRankDatasets/'
    #
    # G: nx.DiGraph = nx.readwrite.read_edgelist(f'{storage_dir}CaliforniaSearch/ca_search_edges.txt', create_using=nx.DiGraph)
    # n = G.number_of_nodes()
    # nodes_df:pd.DataFrame = pd.read_csv(f'{storage_dir}CaliforniaSearch/ca_search_nodes.txt', sep=" ", names=["url"], index_col=0)
    # # print(nodes_df.head(5))
    # node_labels = nodes_df["url"]
    #
    # GraphMatr: np.ndarray = np.array(nx.google_matrix(G)).T
    #
    # # # real_solution: np.ndarray = np.zeros(n, dtype=float)
    # # # pr_dict = nx.pagerank_numpy(G)
    # # # for i, k in enumerate(pr_dict):
    # # #     real_solution[i] = pr_dict[k]
    # # # np.savetxt("ca_search_numpy_sol.txt", real_solution)
    # #
    # real_solution: np.ndarray = np.loadtxt("ca_search_numpy_sol.txt")
    #
    # print(f"Test on real solution: {np.max(np.abs(GraphMatr @ real_solution - real_solution))}")
    #
    # end region

    # region Random stochastic matrix
    # n = 1000
    #
    # GraphMatr: np.ndarray = np.random.rand(n, n)
    # GraphMatr / GraphMatr.sum(axis=1)[:, None]
    # real_solution: np.ndarray = None
    # endregion

    # region test small PR problem with known solution
    # GraphMatr: np.ndarray = np.array([
    #     [0, 0, 0, 1],
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [1, 1, 1, 0],
    # ], dtype=np.float64)
    #
    # GraphMatr: np.ndarray = np.array([
    #     [0,   0,   0,  1],
    #     [0.5, 0,   0,  0],
    #     [0,   0.5, 0,  0],
    #     [0.5, 0.5, 1,  0],
    # ], dtype=np.float64)
    #
    # n = GraphMatr.shape[0]
    #
    # M = GraphMatr
    #
    # # B = np.ones_like(GraphMatr)/n
    # # M = 0.85 * GraphMatr + 0.15 * B
    # # GraphMatr = M
    # # vals, vects = np.linalg.eig(M)
    # #
    # # print("Eigenvects:")
    # # print(f"{vects}")
    # # for i, v in enumerate(vals):
    # #     print(f"Test {i}: {v}: {M @ vects[:, i] - vals[i] * vects[:, i]}")
    # #
    # # real_solution = np.array([0.343, 0.183, 0.115, 0.359])
    # real_solution = np.array([0.36363636, 0.18181818, 0.09090909, 0.36363636])
    #
    # print("Test:")
    # print(M @ real_solution - real_solution)

    # endregion

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 1
    algorithm_params.stop_by = StopCondition.STEP_SIZE

    algorithm_params.save_history = False
    algorithm_params.show_plots = True

    algorithm_params.eps = 1e-5
    algorithm_params.max_iters = 5000
    algorithm_params.min_iters = 10

    algorithm_params.lam = 0.1
    # for Bregman variants
    algorithm_params.lam_KL = 3.0

    # algorithm_params.lam_spec = {'Tseng': 0.1}
    # algorithm_params.lam_spec_KL = {'Tseng': 3.0}

    # algorithm_params.lam_KL = 3. # For MT and EFP

    # algorithm_params.x_limits = [-0.1, 10.]
    # algorithm_params.y_limits = [0.02, 0.5]

    algorithm_params.start_adaptive_lam = 2.5
    algorithm_params.start_adaptive_lam1 = algorithm_params.start_adaptive_lam

    algorithm_params.adaptive_tau = 0.5 * 0.75
    algorithm_params.adaptive_tau_small = 0.33 * 0.75

    algorithm_params.x0 = np.concatenate((np.array([1. / n for i in range(n)]), np.array([1. / n for i in range(n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    # algorithm_params.y_label = "Gap"
    # algorithm_params.x_label = "sec."
    # algorithm_params.y_limits = [1e-3,10]

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 0

    res = PageRankProblem(
        GraphMatr=GraphMatr,
        x0=algorithm_params.x0,
        x_test=real_solution,
        node_labels=node_labels,
        hr_name='$ PageRank ' +
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                f", \ \\lambda_{{KL}} = {round(algorithm_params.lam_KL, 5)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )

    if real_solution is not None:
        print(f"Goal function on real solution: {res.F(real_solution)}")
        print(f"Top 50 ranks: {np.argsort(real_solution)[::-1][:50]}")

    return res
