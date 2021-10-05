import numpy as np
import cvxpy as cp

from constraints.l1_ball import L1Ball
from methods.algorithm_params import AlgorithmParams
from problems.sle_saddle import SLESaddle
from utils.graph.alg_stat_grapher import YAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 5
    m = 10

    M = np.random.rand(n, m) * 3.

    unconstrained_solution = np.array([1. for i in range(m)])

    p = M @ unconstrained_solution

    algorithm_params.x0 = np.array([0.2 for i in range(m)])
    algorithm_params.x1 = np.array([0.1 for i in range(m + n)])

    # norm = np.linalg.norm(M, 2)
    # algorithm_params.lam = 1./norm
    algorithm_params.lam = 0.02

    algorithm_params.start_adaptive_lam = 0.5
    algorithm_params.start_adaptive_lam1 = 0.5

    algorithm_params.max_iters = 2000
    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    # L1 ball radius
    c = 1.5
    # c = 1.34
    constraints = L1Ball(m, c)

    projected_solution = constraints.project(unconstrained_solution)
    print("M:")
    print(M)
    print(f"P: {p}")
    print(f"Projected solution: {projected_solution}; c: {c}")
    print(f"Goal F on proj. sol.: {np.linalg.norm(M @ projected_solution - p)}")
    print()

    x = cp.Variable(m)
    objective = cp.Minimize(cp.sum_squares(M @ x - p))
    constraints_cp = [cp.norm(x, 1) <= c]
    prob = cp.Problem(objective, constraints_cp)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    test_solution = x.value

    print("Solved by CP:")
    print(test_solution)
    print(f"Goal F on CP solution: {np.linalg.norm(M @ test_solution - p)}")
    print(f"CP solution is in C: {constraints.isIn(test_solution)}")
    print()

    return SLESaddle(
        M=M, p=p,
        C=constraints,
        x0=algorithm_params.x0,
        x_test=test_solution,
        hr_name=f"$||Mx - p||_2 \\to min, M_{{ {n}x{m} }}, random, min-max \ form, \ ||x|| \\leq {c} \ \lambda = " +
                str(round(algorithm_params.lam, 3)) + f", \ \\tau = {algorithm_params.adaptive_tau}$"
    )
