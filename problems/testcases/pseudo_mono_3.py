import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):

    algorithm_params.x0 = np.array([-10., 10., -10.])
    # algorithm_params.x0 = np.array([-1., -1., -1.])
    algorithm_params.x1 = algorithm_params.x0.copy()

    algorithm_params.lam = 0.9 / 5.07
    algorithm_params.lam_small = algorithm_params.lam / 2
    # algorithm_params.lam = 1.0/5.07/4.0
    # algorithm_params.lam = 0.01

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    # algorithm_params.adaptive_tau = 0.35
    # algorithm_params.adaptive_tau_large = 0.6

    algorithm_params.adaptive_tau = 0.45
    algorithm_params.adaptive_tau_large = 0.9

    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.x_axis_type = XAxisType.ITERATION

    algorithm_params.y_axis_type = YAxisType.REAL_ERROR

    hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
    hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

    constraints = ConvexSetsIntersection([hr, hp])

    return PseudoMonotoneOperOne(
        C=constraints,
        x0=algorithm_params.x0,
        hr_name='$Ax=f(x)(Mx+p), p = 0, M - 3x3 \ matrix, C = [-5,5]^3 \\times \{x_1+x_2+x_3 = 0\} ' +
                f", \ \\lambda = {round(algorithm_params.lam, 3)}" +
                f", \ \\lambda_2 = {round(algorithm_params.lam_small, 3)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_2 = {round(algorithm_params.adaptive_tau_large, 3)}" +
                '$'
    )
