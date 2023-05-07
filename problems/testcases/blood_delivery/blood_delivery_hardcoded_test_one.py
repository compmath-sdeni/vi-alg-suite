import numpy as np

from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.nagurna_simplest import BloodDeliveryHardcodedOne
from utils.graph.alg_stat_grapher import YAxisType, XAxisType

def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    N = 1
    def_lam = 0.02

    algorithm_params.x0 = np.array([0.1])
    algorithm_params.x1 = algorithm_params.x0.copy()

    # not needed - set in problem
    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.lam = def_lam
    algorithm_params.lam_KL = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = 0.02
    algorithm_params.start_adaptive_lam1 = 0.02

    algorithm_params.adaptive_tau = 0.5 * 0.9
    algorithm_params.adaptive_tau_small = 0.33 * 0.9

    algorithm_params.max_iters = 4000
    algorithm_params.min_iters = 3

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 100
    algorithm_params.stop_by = StopCondition.STEP_SIZE

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-6

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR

    algorithm_params.plot_start_iter = 0
    algorithm_params.time_scale_divider = 1e+9

    real_solution = np.array([62/42])

    hr = Hyperrectangle(1, [[0, 5]])
    constraints = hr

    problem = BloodDeliveryHardcodedOne(
                C=constraints,
                x0=algorithm_params.x0,
                hr_name='$BloodDelivery simplest {0}D$'.format(N),
                xtest=real_solution
                )

    return problem
