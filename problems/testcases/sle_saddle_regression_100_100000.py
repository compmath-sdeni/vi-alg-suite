import numpy as np
import cvxpy as cp

from constraints.l1_ball import L1Ball
from methods.algorithm_params import AlgorithmParams
from problems.sle_saddle import SLESaddle
from utils.graph.alg_stat_grapher import YAxisType, XAxisType


def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):
    n = 100000
    m = 100

    # M = np.round(np.random.rand(n, m) * 10. - 5, 0)
    # unconstrained_solution = np.round(np.random.rand(m) * 10. - 5, 0)
    #
    # np.savetxt('test_LR_M.txt', M)
    # np.savetxt('test_LR_x.txt', unconstrained_solution)

    M = np.loadtxt('test_LR_M.txt')
    unconstrained_solution = np.loadtxt('test_LR_x.txt')

    p = M @ unconstrained_solution

    test_solution = unconstrained_solution

    # for saddle point, there are should also be n dual variables - their initial values can be added here
    # or they will be auto-added inside the problem constructor
    algorithm_params.x0 = unconstrained_solution + np.random.normal(0, 5., unconstrained_solution.shape)

    # algorithm_params.x0 = np.loadtxt('test_LR_x0.txt')
    # np.savetxt('test_LR_x0.txt', algorithm_params.x0)

    # This var is not passed to the problem - so we need to add initial values for dual variables manually.
    # If skipped here, it will be done below - after the problem initialization
    algorithm_params.x1 = algorithm_params.x0

    norm = np.linalg.norm(M, 2)
    algorithm_params.lam = 0.9/norm
    algorithm_params.lam_small = algorithm_params.lam/2

    algorithm_params.start_adaptive_lam = 0.001
    algorithm_params.start_adaptive_lam1 = 0.0005

    # algorithm_params.adaptive_tau = 0.45
    # algorithm_params.adaptive_tau_large = 0.75

    algorithm_params.adaptive_tau = 0.45
    algorithm_params.adaptive_tau_large = 0.9

    algorithm_params.max_iters = 1000
    algorithm_params.y_axis_type = YAxisType.REAL_ERROR
    # algorithm_params.x_axis_type = XAxisType.TIME

    # L1 ball radius
    c = 500
    # c = 1.34
    constraints = L1Ball(m, c)

    print(f"x real in C: {constraints.isIn(unconstrained_solution)}")
    print(f"x0 in C: {constraints.isIn(algorithm_params.x0)}")

    projected_solution = constraints.project(unconstrained_solution)
    print("M:")
    print(M)
    print(f"P: {p}")
    print(f"Projected solution: {projected_solution}; c: {c}")
    print(f"Goal F on proj. sol.: {np.linalg.norm(M @ projected_solution - p)}")
    print()

    # x = cp.Variable(m)
    # objective = cp.Minimize(cp.sum_squares(M @ x - p))
    # constraints_cp = [cp.norm(x, 1) <= c]
    # prob = cp.Problem(objective, constraints_cp)
    # # The optimal objective value is returned by `prob.solve()`.
    # result = prob.solve()
    # # The optimal value for x is stored in `x.value`.
    # test_solution = x.value
    #
    # print("Solved by CP:")
    # print(test_solution)
    # print(f"Goal F on CP solution: {np.linalg.norm(M @ test_solution - p)}")
    # print(f"CP solution is in C: {constraints.isIn(test_solution)}")
    # print()

    problem = SLESaddle(
        M=M, p=p,
        C=constraints,
        x0=algorithm_params.x0,
        x_test=test_solution,
        hr_name=f"$||Mx - p||_2 \\to min, M_{{ {n}x{m} }}, random, min-max \ form, \ ||x|| \\leq {c} " +
                f", \ \\lambda = {round(algorithm_params.lam, 3)}" +
                f", \ \\lambda_2 = {round(algorithm_params.lam_small, 3)}" +
                f", \ \\tau = {round(algorithm_params.adaptive_tau, 3)}" +
                f", \ \\tau_2 = {round(algorithm_params.adaptive_tau_large, 3)}" +
                '$'
    )

    # fix x1 - add the same initial 'y' values, as in x0 (which are added inside problem constructor)
    if algorithm_params.x1.shape[0] == m:
        algorithm_params.x1 = np.concatenate((algorithm_params.x1, problem.x0[m:]))

    return problem
