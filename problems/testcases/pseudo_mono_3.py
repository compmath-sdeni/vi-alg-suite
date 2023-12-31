import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):

    N = 3
    # algorithm_params.x0 = np.array([-10., 10., -10.])  # best for MT-Adapt
    algorithm_params.x0 = np.array([-4., 3., 5.])
    # algorithm_params.x0 = np.array([-1., -1., -1.])
    algorithm_params.x1 = algorithm_params.x0.copy()

    # not needed - set in problem
    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.lam = 0.5 / 5.0679
    algorithm_params.lam_KL = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    algorithm_params.adaptive_tau = 0.5 * 0.9
    algorithm_params.adaptive_tau_small = 0.33 * 0.9

    algorithm_params.max_iters = 400
    algorithm_params.min_iters = 400

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

    hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
    hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

    constraints = ConvexSetsIntersection([hr, hp])

    return PseudoMonotoneOperOne(
        C=constraints,
        x0=algorithm_params.x0,
        hr_name='$Ax=f(x)(Mx+p), p = 0, M - 3x3 \ matrix, C = [-5,5]^3 \\times \{x_1+x_2+x_3 = 0\} ' +
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_KL, 5)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
