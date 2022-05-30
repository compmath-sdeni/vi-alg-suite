import os

import numpy as np
import nashpy as nash
from numpy import inf

from constraints.allspace import Rn
from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.minmax_game import MinMaxGame
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def generateRandomIntDefiniteGame(m: int, n: int, *, a: float = -10, b: float = 10, game_value: float = 0,
                                  eq_row: int = -1, eq_col: int = -1):
    res: np.ndarray = np.round(a + np.random.default_rng().random((m, n)) * (b - a))

    if eq_row < 0:
        eq_row = np.random.randint(0, m)

    if eq_col < 0:
        eq_col = np.random.randint(0, n)

    res[eq_row][eq_col] = game_value

    all_ok = False

    while not all_ok:
        all_ok = True
        min_idx = np.argmin(res[eq_row])
        if min_idx != eq_col:
            all_ok = False
            res[eq_row][min_idx] = game_value + np.random.randint(1, b - game_value + 1)
        max_idx = np.argmax(res[:, eq_col])
        if res[max_idx][eq_col] > game_value:
            all_ok = False
            res[max_idx][eq_col] = game_value - np.random.randint(1, game_value - a + 1)

    all_ok = False
    while not all_ok:
        all_ok = True
        for i in range(0, m):
            min_idx = np.argmin(res[i])
            if res[i, min_idx] > game_value:
                all_ok = False
                res[i, min_idx] = game_value + np.random.randint(1, b - game_value + 1)

        for j in range(0, n):
            max_idx = np.argmax(res[:, j])
            if res[max_idx][j] < game_value:
                all_ok = False
                res[max_idx][j] = game_value - np.random.randint(1, game_value - a + 1)

    return (res, eq_row, eq_col, game_value)


def generateRandomFloatDefiniteGame(m: int, n: int, *, a: float = -10, b: float = 10, game_value: float = 0,
                                    eq_row: int = -1, eq_col: int = -1):
    res: np.ndarray = a + np.random.default_rng().random((m, n)) * (b - a)

    if eq_row < 0:
        eq_row = np.random.randint(0, m)

    if eq_col < 0:
        eq_col = np.random.randint(0, n)

    res[eq_row][eq_col] = game_value

    delta = (b - a) / 10

    all_ok = False
    while not all_ok:
        all_ok = True
        min_idx = np.argmin(res[eq_row])
        if min_idx != eq_col:
            all_ok = False
            res[eq_row][min_idx] = game_value + np.random.default_rng().random() * (b - game_value - delta) + delta
        max_idx = np.argmax(res[:, eq_col])
        if res[max_idx][eq_col] > game_value:
            all_ok = False
            res[max_idx][eq_col] = game_value - np.random.default_rng().random() * (game_value - a - delta) - delta

    all_ok = False
    while not all_ok:
        all_ok = True
        for i in range(0, m):
            min_idx = np.argmin(res[i])
            if res[i, min_idx] > game_value:
                all_ok = False
                res[i, min_idx] = game_value + np.random.default_rng().random() * (b - game_value - delta) + delta

        for j in range(0, n):
            max_idx = np.argmax(res[:, j])
            if res[max_idx][j] < game_value:
                all_ok = False
                res[max_idx][j] = game_value - np.random.default_rng().random() * (game_value - a - delta) - delta

    return (res, eq_row, eq_col, game_value)


def generateRandomFloatDefiniteGameTwoStrat(m: int, n: int, *, a: float = -10, b: float = 10, game_value: float = 0,
                                    eq_row: np.ndarray = None, eq_col: np.ndarray = None):
    res: np.ndarray = a + np.random.default_rng().random((m, n)) * (b - a)

    if eq_row < 0:
        eq_row = np.random.randint(0, m)

    if eq_col < 0:
        eq_col = np.random.randint(0, n)

    res[eq_row][eq_col] = game_value

    delta = (b - a) / 10

    all_ok = False
    while not all_ok:
        all_ok = True
        min_idx = np.argmin(res[eq_row])
        if min_idx != eq_col:
            all_ok = False
            res[eq_row][min_idx] = game_value + np.random.default_rng().random() * (b - game_value - delta) + delta
        max_idx = np.argmax(res[:, eq_col])
        if res[max_idx][eq_col] > game_value:
            all_ok = False
            res[max_idx][eq_col] = game_value - np.random.default_rng().random() * (game_value - a - delta) - delta

    all_ok = False
    while not all_ok:
        all_ok = True
        for i in range(0, m):
            min_idx = np.argmin(res[i])
            if res[i, min_idx] > game_value:
                all_ok = False
                res[i, min_idx] = game_value + np.random.default_rng().random() * (b - game_value - delta) + delta

        for j in range(0, n):
            max_idx = np.argmax(res[:, j])
            if res[max_idx][j] < game_value:
                all_ok = False
                res[max_idx][j] = game_value - np.random.default_rng().random() * (game_value - a - delta) - delta

    return (res, eq_row, eq_col, game_value)

def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    m = 1000
    n = 1500

    md = 0.
    dd = 10.0

    P = None
    real_solution = None

    matr_file_name_base: str = f'minmax_rand_float_P_{m}x{n}'
    if matr_file_name_base and os.path.exists(f'{matr_file_name_base}.npy'):
        P = np.load(f'{matr_file_name_base}.npy')
    else:
        # P = np.random.randint(-10, 10, size=(m, n)).astype(float)
        P = (np.random.rand(m, n)) * (dd*2) - dd + md
        # P = np.random.normal(-5., 20., (m, n))

        if matr_file_name_base:
            np.save(matr_file_name_base, P)

    algorithm_params.x0 = np.concatenate((np.array([1. / n for i in range(n)]), np.array([1. / m for i in range(m)])))
    # algorithm_params.x0 = np.array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.])

    algorithm_params.x1 = algorithm_params.x0.copy()

    # algorithm_params.y_limits = [0.05, 15]

    # rps = nash.Game(-P.T)
    # print(rps)
    # eqs = list(rps.support_enumeration())
    # print(eqs)
    # real_solution = np.concatenate((np.array(eqs[0]), np.array(eqs[1])))
    # print(real_solution)

    algorithm_params.test_time = True
    algorithm_params.test_time_count = 10
    algorithm_params.stop_by = StopCondition.GAP

    algorithm_params.excel_history = False
    algorithm_params.save_history = False
    algorithm_params.save_plots = False

    algorithm_params.eps = 0.01
    algorithm_params.max_iters = 3000
    algorithm_params.min_iters = 5

    algorithm_params.lam = 0.5 / np.linalg.norm(P, 2)
    algorithm_params.lam_medium = 0.0  # 0.45 / np.linalg.norm(P, 2)
    # for Bregman variants
    algorithm_params.lam_KL = 0.5 / np.max(np.abs(P))  #  5.9 / (max(abs(np.max(P)), abs(np.min(P))))

    # algorithm_params.x_limits = [-0.1, 10.]
    # algorithm_params.y_limits = [0.02, 0.5]

    algorithm_params.start_adaptive_lam = 0.05 # Euclid versions
    algorithm_params.start_adaptive_lam1 = 0.5 # KL versions

    algorithm_params.adaptive_tau = 0.45
    algorithm_params.adaptive_tau_small = 0.3 # EFP KL best tau = 1.0 ???

    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.GOAL_OF_AVERAGED
    algorithm_params.y_label = "Gap"
    # algorithm_params.x_label = "sec."
    # algorithm_params.y_limits = [1e-3,10]

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

    algorithm_params.plot_start_iter = 0

    return MinMaxGame(
        P=P, C=Rn(n + m),
        x0=algorithm_params.x0,
        x_test=real_solution,
        hr_name='$ min \ max (Px,y) ' +
#                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
#                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                #                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                #                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
