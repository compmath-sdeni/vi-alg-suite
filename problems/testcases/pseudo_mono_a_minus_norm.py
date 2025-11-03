import numpy as np

from constraints.l2_ball import L2Ball
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.pseudomonotone_oper_a_norm_x import PseudoMonotoneOperAMinusNorm
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):

    N = 3
    r = 3
    a = 4.0
    L = 6.0

    algorithm_params.x0 = np.array([-5., 5., -3.])
    algorithm_params.x0 = algorithm_params.x0 / np.linalg.norm(algorithm_params.x0)
    algorithm_params.x1 = algorithm_params.x0.copy()

    real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.lam = 0.25/L
    algorithm_params.adaptive_tau = 0.35

    algorithm_params.start_adaptive_lam = 2.0
    algorithm_params.start_adaptive_lam1 = algorithm_params.start_adaptive_lam

    algorithm_params.lam_KL = algorithm_params.lam / 2.
    algorithm_params.adaptive_tau_small = 0.33 * 0.9

    algorithm_params.max_iters = 100
    algorithm_params.min_iters = 100

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 100
    algorithm_params.stop_by = StopCondition.EXACT_SOL_DIST

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-16

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR

    algorithm_params.plot_start_iter = 0
    algorithm_params.time_scale_divider = 1e+9

    l2b = L2Ball(N, r)
    constraints = l2b

    return PseudoMonotoneOperAMinusNorm(
        arity=N,
        a = a,
        r = r,
        L = L,
        x0=algorithm_params.x0,
        C=constraints,
        hr_name=f'$({a}-\\|x\\|)x, C = \\{{x \\in R^{N} : \\|x\\| \\leq {r} \\}} ' +
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
