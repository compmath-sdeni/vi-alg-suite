import numpy as np

from constraints.classic_simplex import ClassicSimplex
from constraints.hyperrectangle import Hyperrectangle
from constraints.r_plus import RnPlus
from methods.algorithm_params import AlgorithmParams
from problems.harker_test import HarkerTest
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 1000

    # algorithm_params.x0 = np.array([10 if i%2 == 0 else -10 for i in range(n)])
    algorithm_params.x0 = np.array([1. for i in range(n)])
    algorithm_params.x1 = algorithm_params.x0.copy()

    real_solution = np.array([0.0 for i in range(n)])

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    # constraints = Hyperrectangle(n, [(-1, 1) for i in range(n)])
    # constraints = ClassicSimplex(n, 1.)
    constraints = RnPlus(n)

    # ht = HarkerTest(
    #     n,
    #     C=constraints,
    #     x0=algorithm_params.x0,
    #     xtest=real_solution
    # )
#    np.save(f"hphard_M_{n}", ht.AM)
#    np.save(f"hphard_q_{n}", ht.q)

    algorithm_params.save_history = True
    algorithm_params.show_plots = True

    P = np.load(f'hphard_M_{n}.npy')
    q = np.load(f'hphard_q_{n}.npy')

    ht = HarkerTest(
        n,
        matr=P, q=q,
        C=constraints,
        x0=algorithm_params.x0,
        xtest=real_solution
    )

    algorithm_params.lam = 0.9 / ht.norm
    algorithm_params.lam_small = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = 1.5 / ht.norm
    algorithm_params.start_adaptive_lam1 = 1.5 / ht.norm

    # for adaptive malitsky-tam
    algorithm_params.adaptive_tau = 0.9
    # for adaptive tseng
    algorithm_params.adaptive_tau_small = 0.45

    # # for adaptive EFP
    # algorithm_params.adaptive_tau_small = 0.33

    ht.hr_name = f"$ HPHard, \  {constraints}" \
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" \
                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_small, 5)}" \
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" \
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" \
                f"$"

    return ht
