import numpy as np
import matplotlib.pyplot as plt

from constraints.hyperrectangle import Hyperrectangle
from methods.algorithm_params import AlgorithmParams, StopCondition
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def get_uniform_rand_shortage_expectation_func(a: float, b: float):
    return lambda v: 0 if v >= b else ((0.5 * (a + b) - v) if v <= a else 0.5 * (b - v) * (b - v) / (b - a))


def get_uniform_rand_shortage_expectation_derivative(a: float, b: float):
    return lambda v: -1 if v < a else (0 if v > b else (v - b) / (b - a))


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams(), show_network=True):
    N = 1
    def_lam = 0.0002

    n_paths = 24
    algorithm_params.x0 = np.zeros(n_paths)
    algorithm_params.x1 = algorithm_params.x0.copy()

    # not needed - set in problem
    # real_solution = np.array([0.0 for i in range(N)])

    algorithm_params.lam = def_lam
    algorithm_params.lam_KL = algorithm_params.lam / 2

    algorithm_params.start_adaptive_lam = 0.002
    algorithm_params.start_adaptive_lam1 = 0.002

    algorithm_params.adaptive_tau = 0.5 * 0.9
    algorithm_params.adaptive_tau_small = 0.33 * 0.9

    algorithm_params.max_iters = 4000
    algorithm_params.min_iters = 3

    algorithm_params.test_time = False
    algorithm_params.test_time_count = 100
    algorithm_params.stop_by = StopCondition.STEP_SIZE

    algorithm_params.save_history = True
    algorithm_params.save_plots = True

    algorithm_params.eps = 1e-10

    algorithm_params.x_axis_type = XAxisType.ITERATION
    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    algorithm_params.plot_start_iter = 0
    algorithm_params.time_scale_divider = 1e+9

    real_solution = np.zeros(n_paths)

    net = BloodSupplyNetwork(n_C=2, n_B=2, n_Cmp=2, n_S=2, n_D=2, n_R=3, theta=0.0, lam_minus=[100, 100, 100],
                             lam_plus=[0, 0, 0],
                             edges=[(0, 1), (0, 2),
                                    (1, 3), (1, 4), (2, 3), (2, 4),
                                    (3, 5), (4, 6),
                                    (5, 7), (6, 8),
                                    (7, 9), (7, 10), (8, 9), (8, 10),
                                    (9, 11), (9, 12), (9, 13), (10, 11), (10, 12), (10, 13)
                                    ],
                             paths=[
                                 [0, 2, 6, 8, 10, 14],
                                 [0, 2, 6, 8, 10, 15],
                                 [0, 2, 6, 8, 10, 16],

                                 [0, 2, 6, 8, 11, 19],
                                 [0, 2, 6, 8, 11, 18],
                                 [0, 2, 6, 8, 11, 17],

                                 [0, 3, 7, 9, 13, 19],
                                 [0, 3, 7, 9, 13, 18],
                                 [0, 3, 7, 9, 13, 17],

                                 [0, 3, 7, 9, 12, 14],
                                 [0, 3, 7, 9, 12, 15],
                                 [0, 3, 7, 9, 12, 16],

                                 [1, 5, 7, 9, 13, 19],
                                 [1, 5, 7, 9, 13, 18],
                                 [1, 5, 7, 9, 13, 17],

                                 [1, 5, 7, 9, 12, 14],
                                 [1, 5, 7, 9, 12, 15],
                                 [1, 5, 7, 9, 12, 16],

                                 [1, 4, 6, 8, 10, 14],
                                 [1, 4, 6, 8, 10, 15],
                                 [1, 4, 6, 8, 10, 16],

                                 [1, 4, 6, 8, 11, 19],
                                 [1, 4, 6, 8, 11, 18],
                                 [1, 4, 6, 8, 11, 17]
                             ],
                             c=[
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0),
                                 (lambda f: 0 * f + 0, lambda f: 0)
                             ],
                             z=[
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0),
                                 (lambda f: 0, lambda f: 0)
                             ],

                             r=[
                                 (lambda f: 2 * f, lambda f: 2),
                                 (lambda f: 1.5 * f, lambda f: 1.5)
                             ],

                             # E(t) - expected value, E'(t) - derivative of expected value
                             expected_shortage=[
                                 (
                                     get_uniform_rand_shortage_expectation_func(5, 10),  # E(Delta-)
                                     get_uniform_rand_shortage_expectation_derivative(5, 10),  # E'(Delta-)
                                 ),
                                 (
                                     get_uniform_rand_shortage_expectation_func(40, 50),  # E(Delta-)
                                     get_uniform_rand_shortage_expectation_derivative(40, 50),  # E'(Delta-)
                                 ),
                                 (
                                     get_uniform_rand_shortage_expectation_func(25, 40),  # E(Delta-)
                                     get_uniform_rand_shortage_expectation_derivative(25, 40)  # E'(Delta-)
                                 )
                             ],
                             expected_surplus=[
                                 (
                                     lambda t: 0,  # E(Delta+)
                                     lambda t: 0,  # E'(Delta+)
                                 ),
                                 (
                                     lambda t: 0,  # E(Delta+)
                                     lambda t: 0,  # E'(Delta+)
                                 ),
                                 (
                                     lambda t: 0,  # E(Delta+)
                                     lambda t: 0,  # E'(Delta+)
                                 )
                             ],
                             edge_loss=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                             )

    # region plot distributions
    f1  = get_uniform_rand_shortage_expectation_func(5, 10)
    f2 = get_uniform_rand_shortage_expectation_derivative(5, 10)
    x = np.linspace(0, 20, 500)
    y1 = [f1(t) for t in x]
    y2 = [f2(t) for t in x]

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    # endregion

    if show_network:
        x = np.ones(n_paths)
        net.recalc_link_flows_and_demands(x)
        l = net.get_loss(x)
        grad = net.get_loss_grad(x)
        print(f"Loss 1.6: {l}; Grad: {grad}")

        net.plot(show_flows=True)

    problem = BloodSupplyNetworkProblem(network=net,
                                        x0=algorithm_params.x0,
                                        hr_name='$BloodDelivery simplest {0}D$'.format(N),
                                        xtest=real_solution
                                        )

    return problem
