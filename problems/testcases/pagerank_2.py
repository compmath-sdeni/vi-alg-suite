import numpy as np
from numpy import inf
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.page_rank_problem_2sim import PageRankProblem2Simplex
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    node_labels: list = None

    # region NetworkX based random graph

    # # # n = 4
    # # # G: nx.DiGraph = nx.DiGraph([
    # # #     (1, 2), (1, 4),
    # # #     (2, 3),
    # # #     (3, 4),
    # # #     (4, 1)
    # # # ])
    # #
    # n = 5000
    #
    # G: nx.DiGraph = nx.generators.scale_free_graph(n)
    # nx.readwrite.write_edgelist(G, f'pagerank_G_{n}.edges')
    #
    # # G: nx.DiGraph = nx.readwrite.read_edgelist(f'pagerank_G_{n}.edges', create_using=nx.DiGraph)
    # GraphMatr:np.ndarray = np.array(nx.google_matrix(G)).T
    #
    # # pos = nx.spring_layout(G)
    # # nx.draw(G, pos)
    # # plt.show()
    # # exit()
    #
    # real_solution: np.ndarray = np.zeros(n, dtype=float)
    # pr_dict = nx.pagerank_numpy(G)
    # for i, k in enumerate(pr_dict):
    #     real_solution[i] = pr_dict[k]
    #
    # # # print(f"Test PR: {real_solution}")
    # print(f"Test on real solution: {np.max(np.abs(GraphMatr @ real_solution - real_solution))}")
    # endregion

    # region California search edges list
    # # https://www.quora.com/Where-can-I-find-arff-data-sets-for-implementing-page-rank
    # # http://vlado.fmf.uni-lj.si/pub/networks/data/mix/mixed.htm
    # # https://www.cs.cornell.edu/courses/cs685/2002fa/data/gr0.California
    # # Jon M. Kleinberg, Authoritative sources in a hyperlinked environment, Journal of the ACMVolume 46Issue 5Sept. 1999 pp 604–632, https://doi.org/10.1145/324133.324140
    # # 9664 (6175 used) nodes, 16150 edges
    #
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
    # # Use networkx to calculate solution to compare with
    # # real_solution: np.ndarray = np.zeros(n, dtype=float)
    # # pr_dict = nx.pagerank_numpy(G)
    # # for i, k in enumerate(pr_dict):
    # #     real_solution[i] = pr_dict[k]
    # # np.savetxt("ca_search_numpy_sol.txt", real_solution)
    #
    # real_solution: np.ndarray = np.loadtxt(f"{storage_dir}CaliforniaSearch/ca_search_numpy_sol.txt")
    #
    # print(f"Test on real solution: {np.max(np.abs(GraphMatr @ real_solution - real_solution))}")
    # print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

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

    # approx solution using numpy linalg
    # B = np.ones_like(GraphMatr)/n
    # M = 0.85 * GraphMatr + 0.15 * B
    # GraphMatr = M
    # vals, vects = np.linalg.eig(M)
    #
    # print("Eigenvects:")
    # print(f"{vects}")
    # for i, v in enumerate(vals):
    #     print(f"Test {i}: {v}: {M @ vects[:, i] - vals[i] * vects[:, i]}")

    # real_solution = np.array([0.343, 0.183, 0.115, 0.359])
    # real_solution = np.array([0.36363636, 0.18181818, 0.09090909, 0.36363636])

    # print("Test Ax-x:")
    # print(M @ real_solution - real_solution)

    # endregion

  # region bigger test PR problem
  #   GraphMatr: np.ndarray = np.array([
  #       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  #       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  #       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  #       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  #       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  #       [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
  #       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  #       [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
  #       [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
  #       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
  #   ], dtype=np.float64)
  #
  #   n = GraphMatr.shape[0]
  #
  #   # divide every column of GraphMatr by the column sum to get Markov matrix
  #   s = GraphMatr.sum(axis=0)
  #   GraphMatr /= s
  #
  #   B = np.ones_like(GraphMatr) / n
  #   M = 0.85 * GraphMatr + 0.15 * B
  #   GraphMatr = M
  #
  #   real_solution = np.array(
  #       [3.476e-2, 6.430e-2, 6.209e-2, 8.753e-2, 1.092e-1, 1.423e-1, 1.831e-1, 2.325e-1, 4.953e-2, 3.476e-2])
  #
  #   vals, vects = np.linalg.eig(GraphMatr)
  #
  #   print("Eigenvects:")
  #   # print(f"{vects}")
  #   for i, v in enumerate(vals):
  #       # print(f"Test {i}: {v}: {GraphMatr @ vects[:, i] - vals[i] * vects[:, i]}")
  #       if abs(v - 1) < 0.000001:
  #           real_solution = vects[:, i]
  #
  #   real_solution /= real_solution.sum()
  #
  #   print(f"Real solution: {real_solution}")
# endregion

# region test PR problem from PhD thesis
    GraphMatr: np.ndarray = np.array([
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0]
    ], dtype=np.float64)

    n = GraphMatr.shape[0]

    # divide every column of GraphMatr by the column sum to get Markov matrix
    s = GraphMatr.sum(axis=0)
    GraphMatr /= s
    print(GraphMatr)

    B = np.ones_like(GraphMatr) / n
    M = 0.85 * GraphMatr + 0.15 * B
    GraphMatr = M

    real_solution = np.array(
        [3.476e-2, 6.430e-2, 6.209e-2, 8.753e-2, 1.092e-1, 1.423e-1, 1.831e-1, 2.325e-1, 4.953e-2, 3.476e-2])

    vals, vects = np.linalg.eig(GraphMatr)

    print("Eigenvects:")
    # print(f"{vects}")
    for i, v in enumerate(vals):
        # print(f"Test {i}: {v}: {GraphMatr @ vects[:, i] - vals[i] * vects[:, i]}")
        if abs(v - 1) < 0.000001:
            real_solution = vects[:, i]

    real_solution /= real_solution.sum()

    print(f"Real solution: {real_solution}")

# end region

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 1
    algorithm_params.stop_by = StopCondition.GAP

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-18
    algorithm_params.max_iters = 1000
    algorithm_params.min_iters = 100

    algorithm_params.lam = 0.05
    # for Bregman variants
    algorithm_params.lam_KL = 1.0/np.max(np.abs(GraphMatr - np.eye(GraphMatr.shape[0])))

    # algorithm_params.lam_spec = {'Tseng': 0.1}
    # algorithm_params.lam_spec_KL = {'Tseng': 3.0}

    # algorithm_params.lam_KL = 3. # For MT and EFP

    # algorithm_params.x_limits = [-0.1, 10.]
    # algorithm_params.y_limits = [0.02, 0.5]

    algorithm_params.start_adaptive_lam = 2.5
    algorithm_params.start_adaptive_lam1 = algorithm_params.start_adaptive_lam

    algorithm_params.adaptive_tau = 0.5 * 0.75
    algorithm_params.adaptive_tau_small = 0.33 * 0.75

    algorithm_params.x0 = np.concatenate((np.array([1. / n for i in range(n)]), np.array([1. / n for i in range(2*n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    algorithm_params.y_label = "$G(z)$"
    # algorithm_params.x_label = "sec."
    # algorithm_params.y_limits = [1e-3,10]

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 5

    res = PageRankProblem2Simplex(
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
        print(f"Goal function on test solution: {res.F(real_solution)}")
        print(f"Top 50 ranks by test solution: {np.argsort(real_solution)[::-1][:50]}")

    print(f"Goal function on start: {res.F(algorithm_params.x0)}")

    return res


def prepareCaliforniaGraphProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    node_labels: list = None

    # region California search edges list
    # https://www.quora.com/Where-can-I-find-arff-data-sets-for-implementing-page-rank
    # http://vlado.fmf.uni-lj.si/pub/networks/data/mix/mixed.htm
    # https://www.cs.cornell.edu/courses/cs685/2002fa/data/gr0.California
    # Jon M. Kleinberg, Authoritative sources in a hyperlinked environment, Journal of the ACMVolume 46Issue 5Sept. 1999 pp 604–632, https://doi.org/10.1145/324133.324140
    # 9664 (6175 used) nodes, 16150 edges

    storage_dir: str = '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/data/PageRankDatasets/'

    G: nx.DiGraph = nx.readwrite.read_edgelist(f'{storage_dir}CaliforniaSearch/ca_search_edges.txt', create_using=nx.DiGraph)
    n = G.number_of_nodes()
    nodes_df:pd.DataFrame = pd.read_csv(f'{storage_dir}CaliforniaSearch/ca_search_nodes.txt', sep=" ", names=["url"], index_col=0)
    # print(nodes_df.head(5))
    node_labels = nodes_df["url"]

    GraphMatr: np.ndarray = np.array(nx.google_matrix(G)).T

    # Use networkx to calculate solution to compare with
    # real_solution: np.ndarray = np.zeros(n, dtype=float)
    # pr_dict = nx.pagerank_numpy(G)
    # for i, k in enumerate(pr_dict):
    #     real_solution[i] = pr_dict[k]
    # np.savetxt("ca_search_numpy_sol.txt", real_solution)

    real_solution: np.ndarray = np.loadtxt(f"{storage_dir}CaliforniaSearch/ca_search_numpy_sol.txt")

    print(f"Test on real solution: {np.max(np.abs(GraphMatr @ real_solution - real_solution))}")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # end region

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 1
    algorithm_params.stop_by = StopCondition.STEP_SIZE

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-15
    algorithm_params.max_iters = 20000
    algorithm_params.min_iters = 15

    algorithm_params.lam = 0.05
    # for Bregman variants
    algorithm_params.lam_KL = 1.0/np.max(np.abs(GraphMatr - np.eye(GraphMatr.shape[0])))

    # algorithm_params.lam_spec = {'Tseng': 0.1}
    # algorithm_params.lam_spec_KL = {'Tseng': 3.0}

    # algorithm_params.lam_KL = 3. # For MT and EFP

    # algorithm_params.x_limits = [-0.1, 10.]
    # algorithm_params.y_limits = [0.02, 0.5]

    algorithm_params.start_adaptive_lam = 2.5
    algorithm_params.start_adaptive_lam1 = algorithm_params.start_adaptive_lam

    algorithm_params.adaptive_tau = 0.5 * 0.75
    algorithm_params.adaptive_tau_small = 0.33 * 0.75

    algorithm_params.x0 = np.concatenate((np.array([1. / n for i in range(n)]), np.array([1. / n for i in range(2*n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    algorithm_params.y_label = "$G(z_n)$"
    # algorithm_params.x_label = "sec."
    # algorithm_params.y_limits = [1e-3,10]

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 0

    res = PageRankProblem2Simplex(
        GraphMatr=GraphMatr,
        x0=algorithm_params.x0,
        x_test=real_solution,
        node_labels=node_labels,
        hr_name='$ PageRank2S ' +
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                f", \ \\lambda_{{KL}} = {round(algorithm_params.lam_KL, 5)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )

    if real_solution is not None:
        print(f"Goal function on test solution: {res.F(real_solution)}")
        print(f"Top 50 ranks by test solution: {np.argsort(real_solution)[::-1][:50]}")

    print(f"Goal function on start: {res.F(algorithm_params.x0)}")

    return res
