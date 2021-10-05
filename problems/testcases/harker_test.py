import numpy as np

from constraints.classic_simplex import ClassicSimplex
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.harker_test import HarkerTest
from utils.graph.alg_stat_grapher import YAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 5

    algorithm_params.x0 = np.array([0.5 for i in range(n)])
    algorithm_params.x1 = np.array([0.25 for i in range(n)])

    real_solution = np.array([0.0 for i in range(n)])
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR

    constraints = Hyperrectangle(n, [(-5, 5) for i in range(n)])
    # constraints = ClassicSimplex(n, n)

    algorithm_params.adaptive_tau = 0.5

    ht = HarkerTest(
        n,
        C=constraints,
        x0=algorithm_params.x0,
        xtest=real_solution,
        hr_name=f"$ HPHard, \ -5 \\leq x_i  \\leq 5 \ \lambda = " +
                str(round(algorithm_params.lam, 3)) + f", \ \\tau = {algorithm_params.adaptive_tau}"
                + f", \ \\tau_2 = {algorithm_params.adaptive_tau_large}$"
    )

    algorithm_params.lam = 0.4 / ht.norm
    algorithm_params.start_adaptive_lam = 1.5
    algorithm_params.start_adaptive_lam1 = 1.5


    return ht
