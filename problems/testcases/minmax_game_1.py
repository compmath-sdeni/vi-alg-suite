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

    delta = (b-a)/10

    all_ok = False
    while not all_ok:
        all_ok = True
        min_idx = np.argmin(res[eq_row])
        if min_idx != eq_col:
            all_ok = False
            res[eq_row][min_idx] = game_value + np.random.default_rng().random()*(b - game_value - delta) + delta
        max_idx = np.argmax(res[:, eq_col])
        if res[max_idx][eq_col] > game_value:
            all_ok = False
            res[max_idx][eq_col] = game_value - np.random.default_rng().random()*(game_value - a - delta) - delta

    all_ok = False
    while not all_ok:
        all_ok = True
        for i in range(0, m):
            min_idx = np.argmin(res[i])
            if res[i, min_idx] > game_value:
                all_ok = False
                res[i, min_idx] = game_value + np.random.default_rng().random()*(b - game_value - delta) + delta

        for j in range(0, n):
            max_idx = np.argmax(res[:, j])
            if res[max_idx][j] < game_value:
                all_ok = False
                res[max_idx][j] = game_value - np.random.default_rng().random()*(game_value - a - delta) - delta

    return (res, eq_row, eq_col, game_value)


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    # region Random problem nxm
    # n = 1000
    # m = 1000
    # #P = np.random.randint(-3, 4, size=(m,n)).astype(float)
    # # P = np.random.rand(m, n)
    # # P = np.random.normal(0., 10., (m, n))
    # real_solution = None
    #
    # #np.save('minmax_P_1000_symmetric_-3_3', P)
    # P = np.load('minmax_P_1000_symmetric_-3_3.npy')
    #
    # algorithm_params.x0 = np.concatenate((np.array([1. / m for i in range(m)]), np.array([1. / n for i in range(n)])))
    # algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion

    # region Test problem one - 3x3
    # n = 3
    # m = 3
    # P = np.array([[1,-1,-1], [-1,-1,3],[-1,3,-1]])
    # real_solution = np.array([0.5, 0.25, 0.25, 0.5, 0.25, 0.25])
    #
    # algorithm_params.x0 = np.concatenate((np.array([1./m for i in range(m)]), np.array([1./n for i in range(n)])))
    # algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion

    # region Test random fully defined game with known solution
    n = 3
    m = 5
    eq_row = 2
    eq_col = n-2
    game_val = 0

    A, eq_row, eq_col, game_val = generateRandomFloatDefiniteGame(m, n, a=-50, b=50,
                                                                game_value=game_val, eq_row=eq_row, eq_col=eq_col)
    print(f"eq_row: {eq_row}; eq_col: {eq_col}; game_val: {game_val}; P:\n{A}")
    P = -A.transpose()

    np.save(f'matrix_game_float_P_{m}x{n}_gv={game_val}_i={eq_row}_j={eq_col}', P)
    # P = np.load(f'matrix_game_float_P_{m}x{n}_gv={game_val}_i={eq_row}_j={eq_col}.npy')

    real_solution = np.zeros(m + n)
    real_solution[eq_row] = 1.
    real_solution[m + eq_col] = 1.

    algorithm_params.x0 = np.concatenate((np.array([1. / m for i in range(m)]), np.array([1. / n for i in range(n)])))
    algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion


    # region Test Blotto game (non-zero value, 4x5)
    # # max_x(min_y((x,Ay)) <=> min_x(max_y(-A^Tx,y))
    # n = 4
    # m = 5
    # P = -np.array([
    #     [4, 2, 1, 0],
    #     [1, 3, 0, -1],
    #     [-2, 2, 2, -2],
    #     [-1, 0, 3, 1],
    #     [0, 1, 2, 4],
    # ]).transpose()
    #
    # # Game value: 1.55556
    # # Not unique!
    # real_solution = np.array([0.44444, 0, 0.11111, 0, 0.44444,
    #                           0.03333,0.53333,0.35556,0.07778])
    # algorithm_params.x0 = np.concatenate((np.array([1./m for i in range(m)]), np.array([1./n for i in range(n)])))
    # algorithm_params.x1 = algorithm_params.x0.copy()
    # endregion

    # rps = nash.Game(-P.T)
    # print(rps)
    # eqs = list(rps.support_enumeration())
    # print(eqs)
    #    real_solution = np.concatenate((np.array(eqs[0]), np.array(eqs[1])))
    #    print(real_solution)

    algorithm_params.eps = 1e-5
    algorithm_params.max_iters = 5000

    algorithm_params.lam = 0.9 / np.linalg.norm(P, 2)
    algorithm_params.lam_small = 0.45 / np.linalg.norm(P, 2)

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    algorithm_params.adaptive_tau = 0.95
    algorithm_params.adaptive_tau_small = 0.45

    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR
    algorithm_params.y_label = "duality gap"

    algorithm_params.time_scale_divider = 1e+9
    # algorithm_params.x_label = "Time, sec."

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
