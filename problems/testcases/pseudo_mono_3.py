import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    N = 3
    algorithm_params.x0 = np.array([2., -5., 3.])
    algorithm_params.x1 = np.array([3., -2., -1.])
    # algorithm_params.lam = 1.0/5.07/4.0
    # algorithm_params.lam = 0.01
    algorithm_params.lam = 1.0 / 5.07
    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    real_solution = np.array([0.0 for i in range(N)])

    hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
    hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

    constraints = ConvexSetsIntersection([hr, hp])

    return PseudoMonotoneOperOne(
        C=constraints,
        x0=algorithm_params.x0,
        hr_name='$Ax=f(x)(Mx+p), p = 0, M - 3x3 \ matrix, C = [-5,5]^3 \\times \{x_1+x_2+x_3 = 0\}, \ \lambda = ' + str(
            round(algorithm_params.lam, 3)) + '$'
    )
