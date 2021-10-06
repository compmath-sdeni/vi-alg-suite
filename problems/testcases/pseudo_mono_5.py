import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.halfspace import HalfSpace
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.pseudomonotone_oper_two import PseudoMonotoneOperTwo
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    algorithm_params.x0 = np.array([-10., 10., -10., 10., -10.])
    algorithm_params.x1 = algorithm_params.x0.copy()

    # algorithm_params.lam = 0.2
    # algorithm_params.lam = 0.02
    # algorithm_params.lam = 1.0/5.07
    algorithm_params.lam = 0.18
    algorithm_params.lam_small = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    # algorithm_params.adaptive_tau = 0.35
    # algorithm_params.adaptive_tau_large = 0.65

    algorithm_params.adaptive_tau = 0.45
    algorithm_params.adaptive_tau_large = 0.9

    algorithm_params.real_solution = np.array([0.28484841, -0.60606057, -0.8303029, 0.36363633, 0.31515152])
    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    hr = Hyperrectangle(5, [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]])
    hp = HalfSpace(a=np.array([1., 1., 1., 1., 1.]), b=5.)

    constraints = ConvexSetsIntersection([hr, hp])

    return PseudoMonotoneOperTwo(
        C=constraints,
        x0=algorithm_params.x0,
        x_test=algorithm_params.real_solution,
        hr_name='$Ax=f(x)(Mx+p), p \\ne 0, M - 5x5 \ matrix, \ C = [-5,5]^5 \\times \{x_1 + ... +x_5 <= 5\} ' +
                f", \ \\lambda = {round(algorithm_params.lam, 3)}" +
                f", \ \\lambda_2 = {round(algorithm_params.lam_small, 3)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_2 = {round(algorithm_params.adaptive_tau_large, 3)}" +
                '$'
    )
