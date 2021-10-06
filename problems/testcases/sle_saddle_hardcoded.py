import numpy as np
import cvxpy as cp

from constraints.l1_ball import L1Ball
from methods.algorithm_params import AlgorithmParams
from problems.sle_saddle import SLESaddle
from utils.graph.alg_stat_grapher import YAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 2  # rows (observations)
    m = 3  # columns (variables)

    M = np.array([
        [5, 2, 1]
        , [2, 13, 4]
    ], dtype=float)


    unconstrained_solution = np.array([1, 1, 1])

    algorithm_params.x0 = np.array([0.2 for i in range(m)])
    algorithm_params.x1 = np.array([0.1 for i in range(m + n)])

    # set p by wanted solution
    p = M @ unconstrained_solution

    norm = np.linalg.norm(M, 2)  # Can be used to get L constant
    # algorithm_params.lam = 1./norm
    algorithm_params.lam = 0.02

    algorithm_params.start_adaptive_lam = 0.5
    algorithm_params.start_adaptive_lam1 = 0.5

    algorithm_params.adaptive_tau = 0.4

    algorithm_params.max_iters = 10000

    algorithm_params.y_axis_type = YAxisType.STEP_DELTA

    c = 10. # L1 ball radius
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
        hr_name='$||Mx - p||_2 \\to min, M_{2x3}, predefined, min-max \ form, \ ||x|| \\leq ' + str(c) + ' \ \lambda = ' +
                str(round(algorithm_params.lam, 3)) + ', \ \\tau = ' + str(algorithm_params.adaptive_tau) + '$'
    )
