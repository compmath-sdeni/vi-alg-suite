import numpy as np

from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.nagurna_simplest import BloodBankingOne
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
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
    algorithm_params.stop_by = StopCondition.EXACT_SOL_DIST

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-16

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR

    algorithm_params.plot_start_iter = 0
    algorithm_params.time_scale_divider = 1e+9

    real_solution = np.array([62 / 42])

    hr = Hyperrectangle(1, [[0, 5]])
    constraints = hr

    net = BloodSupplyNetwork(n_C=1, n_B=1, n_Cmp=1, n_S=1, n_D=1, n_R=1, theta=1, lam_minus=100, lam_plus=0,
                             edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
                             paths=[
                                 [0, 1, 2, 3, 4, 5]
                                    ],
                             c=[
                                 (lambda f: f + 6, lambda f: 1),
                                 (lambda f: 2 * f + 7, lambda f: 2),
                                 (lambda f: f + 11, lambda f: 1),
                                 (lambda f: 3 * f + 11, lambda f: 3),
                                 (lambda f: f + 2, lambda f: 1),
                                 (lambda f: f + 1, lambda f: 1)],

                             z=[
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0)
                             ],

                             r=[(lambda f: 2 * f, lambda f: 2)],

                             # E(t) - expected value, E'(t) - derivative of expected value
                             shortage=[
                                 (
                                     lambda t: 0 if t <= 0 else (2.5 - t + t*t/2.0 if t < 5 else 0), # E(Delta-)
                                     lambda t: 0 if t <= 0 else (t/5 - 1 if t < 5 else 0),  # E'(Delta-)
                                 )],

                             surplus=[
                                 (
                                     lambda t: 0,  # E(Delta+)
                                     lambda t: 0,  # E'(Delta+)
                                 )]
                             )

    problem = BloodSupplyNetworkProblem(network=net,
        x0=algorithm_params.x0,
        hr_name='$BloodDelivery simplest {0}D$'.format(N),
        xtest=real_solution
    )

    return problem
