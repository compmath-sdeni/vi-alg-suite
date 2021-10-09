import numpy as np

from constraints.classic_simplex import ClassicSimplex
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.harker_test import HarkerTest
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 100

    algorithm_params.x0 = np.array([10 if i%2 == 0 else -10 for i in range(n)])
    algorithm_params.x1 = algorithm_params.x0.copy()

    real_solution = np.array([0.0 for i in range(n)])

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    #constraints = Hyperrectangle(n, [(-5, 5) for i in range(n)])
    constraints = ClassicSimplex(n, n)

    ht = HarkerTest(
        n,
        C=constraints,
        x0=algorithm_params.x0,
        xtest=real_solution
    )

    algorithm_params.lam = 0.9 / ht.norm
    algorithm_params.lam_small = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = algorithm_params.lam * 10
    algorithm_params.start_adaptive_lam1 = algorithm_params.lam * 10

    # for adaptive malitsky-tam
    algorithm_params.adaptive_tau = 0.65
    # for adaptive tseng
    algorithm_params.adaptive_tau_small = 0.45

    # for adaptive EFP
    algorithm_params.adaptive_tau_small = 0.33

    ht.hr_name = f"$ HPHard, \ -5 \\leq x_i  \\leq 5 \\times {n}d simplex " \
    f", \ \\lambda = {round(algorithm_params.lam, 3)}" \
    f", \ \\lambda_2 = {round(algorithm_params.lam_small, 3)}" \
    f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" \
    f", \ \\tau_2 = {round(algorithm_params.adaptive_tau_small, 3)}" \
    f"$"

    return ht
