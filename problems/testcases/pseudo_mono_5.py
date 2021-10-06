import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.halfspace import HalfSpace
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.pseudomonotone_oper_two import PseudoMonotoneOperTwo
from utils.graph.alg_stat_grapher import YAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    algorithm_params.x0 = np.array([2., -5., 3., -1., 2.])
    algorithm_params.x1 = np.array([2.5, -4., 2., -1.5, 2.5])

    # algorithm_params.lam = 0.2
    # algorithm_params.lam = 0.02
    # algorithm_params.lam = 1.0/5.07
    algorithm_params.lam = 0.13

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    algorithm_params.adaptive_tau = 0.25
    algorithm_params.adaptive_tau_large = 0.5

    algorithm_params.real_solution = np.array([0.28484841, -0.60606057, -0.8303029, 0.36363633, 0.31515152])
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR


    hr = Hyperrectangle(5, [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]])
    hp = HalfSpace(a=np.array([1., 1., 1., 1., 1.]), b=5.)

    constraints = ConvexSetsIntersection([hr, hp])

    return PseudoMonotoneOperTwo(
        C=constraints,
        x0=algorithm_params.x0,
        x_test=algorithm_params.real_solution,
        hr_name='$Ax=f(x)(Mx+p), p \\ne 0, M - 5x5 \ matrix, \ C = [-5,5]^5 \\times \{x_1 + ... +x_5 <= 5\}, \ \lambda = ' + str(
            round(algorithm_params.lam, 3)) + ', \ \\tau = ' + str(algorithm_params.adaptive_tau) + '$'
    )
