import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):

    # algorithm_params.x0 = np.array([-10., 10., -10.])  # best for MT-Adapt
    algorithm_params.x0 = np.array([-4., 3., 5.])
    # algorithm_params.x0 = np.array([-1., -1., -1.])
    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.lam = 0.9 / 5.0679
    algorithm_params.lam_small = algorithm_params.lam / 2
    # algorithm_params.lam = 1.0/5.07/4.0
    # algorithm_params.lam = 0.01

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    # algorithm_params.adaptive_tau = 0.6
    # algorithm_params.adaptive_tau_small = 0.35

    algorithm_params.adaptive_tau = 0.65
    algorithm_params.adaptive_tau_small = 0.45

    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.eps = 1e-8

    algorithm_params.x_axis_type = XAxisType.TIME
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR

    hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
    hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

    constraints = ConvexSetsIntersection([hr, hp])

    return PseudoMonotoneOperOne(
        C=constraints,
        x0=algorithm_params.x0,
        hr_name='$Ax=f(x)(Mx+p), p = 0, M - 3x3 \ matrix, C = [-5,5]^3 \\times \{x_1+x_2+x_3 = 0\} ' +
                f", \ \\lambda = {round(algorithm_params.lam, 5)}" +
                f", \ \\lambda_{{small}} = {round(algorithm_params.lam_small, 5)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_{{small}} = {round(algorithm_params.adaptive_tau_small, 3)}" +
                '$'
    )
