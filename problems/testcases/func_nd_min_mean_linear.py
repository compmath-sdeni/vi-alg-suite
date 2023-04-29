import numpy as np

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.funcndmin import FuncNDMin
from utils.graph.alg_stat_grapher import YAxisType, XAxisType

def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):

    N = 10
    def_lam = 1. / (4. * N)

    # algorithm_params.x0 = np.array([-10., 10., -10.])  # best for MT-Adapt
    algorithm_params.x0 = np.array([-4., 3., 5., -4., 3., 5., -4., 3., 5., -4.])
    # algorithm_params.x0 = np.array([-1., -1., -1.])
    algorithm_params.x1 = algorithm_params.x0.copy()

    # not needed - set in problem
    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.lam = def_lam
    algorithm_params.lam_KL = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = 1.
    algorithm_params.start_adaptive_lam1 = 1.

    algorithm_params.adaptive_tau = 0.5 * 0.9
    algorithm_params.adaptive_tau_small = 0.33 * 0.9

    algorithm_params.max_iters = 400
    algorithm_params.min_iters = 3

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 100
    algorithm_params.stop_by = StopCondition.EXACT_SOL_DIST

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-16

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    algorithm_params.plot_start_iter = 0
    algorithm_params.time_scale_divider = 1e+9

    x0 = np.array([i + 1 for i in range(N)])
    x1 = np.array([i + 0.5 for i in range(N)])

    real_solution = np.array([0.5 for i in range(N)])

    hr = Hyperrectangle(N, [[-1, 1] for i in range(N)])
    constraints = hr

    # hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
    # hp = Hyperplane(a=np.array([1., 1., 1.]), b=3./2.)
    # constraints = ConvexSetsIntersection([hr, hp])

    problem = FuncNDMin(N,
                lambda x: (np.sum(x) - N / 2) ** 2,
                lambda x: np.ones(N) * 2 * (np.sum(x) - N / 2),
                C=constraints,
                x0=algorithm_params.x0,
                hr_name='$(x_1 + x_2 + ... + x_n - n/2)^2->min, C = [-5,5]x[-5,5], N = {0}$'.format(N),
                xtest=real_solution
                )

    return problem
