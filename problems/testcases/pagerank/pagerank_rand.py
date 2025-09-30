import numpy as np
from numpy import inf
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from constraints.allspace import Rn
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.page_rank_problem import PageRankProblem
from utils.graph.alg_stat_grapher import YAxisType, XAxisType

def prepareProblem(n, *, algorithm_params: AlgorithmParams = AlgorithmParams()):
    probl = PageRankProblem.CreateRandom(n, 0.1)
    GraphMatr = probl.M

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 1
    algorithm_params.stop_by = StopCondition.STEP_SIZE

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-18
    algorithm_params.max_iters = 6000
    algorithm_params.min_iters = 500

    algorithm_params.lam = 0.1
    # for Bregman variants
    algorithm_params.lam_KL = 1.0/np.max(np.abs(GraphMatr - np.eye(GraphMatr.shape[0])))
    algorithm_params.lam_KL = 0.1

    algorithm_params.start_adaptive_lam = 1.0
    algorithm_params.start_adaptive_lam1 = algorithm_params.start_adaptive_lam

    algorithm_params.adaptive_tau = 0.01 # np.sqrt(2.0) - 1 # 0.5 * 0.75
    algorithm_params.adaptive_tau_small = 0.1# 0.33 * 0.75

    algorithm_params.x0 = np.concatenate((np.array([1. / n for i in range(n)]), np.array([1. / n for i in range(n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.moving_average_window = 100
    algorithm_params.result_averaging_window = 200

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_FUNCTION
    # algorithm_params.y_label = "$\lambda_n$"
    algorithm_params.y_label = "$G({x_n})$"
    # algorithm_params.x_label = "sec."
    # algorithm_params.y_limits = [1e-3,10]

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 5

    test_solution = None
    eValues, eVectors = np.linalg.eig(probl.M)
    for i, v in enumerate(eValues):
        if abs(v - 1) < 0.00000001:
            v = abs(eVectors[:, i])
            v /= v.sum()
            test_solution = np.copy(v)
            print("Eigenvector for 1: ", v)
            print("GAP on test solution: ", np.linalg.norm(GraphMatr @ test_solution - test_solution))


    probl._x0 = algorithm_params.x0
    probl.x_test = test_solution
    probl.hr_name=f"$ PageRank, \ \\lambda = {round(algorithm_params.lam, 5)}" + f", \ \\lambda_{{KL}} = {round(algorithm_params.lam_KL, 5)}" + f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" + f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" + '$'

    if test_solution is not None:
        print(f"Goal function on test solution: {probl.F(test_solution)}")
        print(f"Top 50 ranks by test solution: {np.argsort(test_solution)[::-1][:50]}")

    print(f"Goal function on start: {probl.F(algorithm_params.x0)}")

    if probl.L is not None:
        print(f"Lipschitz constant: {probl.L}; 1/L: {1.0/probl.L}; 1/2L: {0.5/probl.L}; 1/3L: {1.0/(3*probl.L)}; (sqrt(2)-1)/L: {(np.sqrt(2)-1.0)/probl.L}")
    else:
        print("Lipschitz constant not known")

    return probl