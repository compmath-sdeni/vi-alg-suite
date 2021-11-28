import math
import datetime
import os
import io
import sys

import numpy as np
import cvxpy as cp
import pandas as pd
from matplotlib import pyplot as plt

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.classic_simplex import ClassicSimplex
from constraints.halfspace import HalfSpace
from constraints.hyperplane import Hyperplane
from constraints.l1_ball import L1Ball
from methods.IterGradTypeMethod import ProjectionType
from methods.algorithm_params import AlgorithmParams
from methods.extrapol_from_past import ExtrapolationFromPast
from methods.extrapol_from_past_adaptive import ExtrapolationFromPastAdapt
from methods.korpele_mod import KorpelevichMod
from methods.malitsky_tam import MalitskyTam
from methods.malitsky_tam_adaptive import MalitskyTamAdaptive
from methods.tseng import Tseng
from methods.tseng_adaptive import TsengAdaptive
from problems.harker_test import HarkerTest
from problems.matrix_oper_vi import MatrixOperVI
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from problems.pseudomonotone_oper_two import PseudoMonotoneOperTwo
from problems.sle_direct import SLEDirect
from problems.sle_saddle import SLESaddle

from problems.testcases import pseudo_mono_3, pseudo_mono_5, sle_saddle_hardcoded, sle_saddle_random_one, harker_test, \
    sle_saddle_regression_100_100000, minmax_game_1

from problems.testcases.transport import pigu_sample, braess_sample, load_file_sample

from problems.testcases.slar_random import getSLE
from utils.alg_history import AlgHistory
from utils.graph.alg_stat_grapher import AlgStatGrapher, XAxisType, YAxisType

from constraints.hyperrectangle import Hyperrectangle

from methods.korpelevich import Korpelevich
from problems.funcndmin import FuncNDMin
from utils.test_alghos import BasicAlgoTests

params = AlgorithmParams(
    eps=1e-5,
    min_iters=10,
    max_iters=2000,
    lam=0.01,
    lam_KL=0.005,
    start_adaptive_lam=0.5,
    start_adaptive_lam1=0.5,
    adaptive_tau=0.75,
    adaptive_tau_small=0.45
)

captured_io = io.StringIO()
sys.stdout = captured_io

# region Simple 2d func min

# f_to_min = lambda x: (x[0]-2) ** 2 + (x[1]+1) ** 2
# C = Hyperrectangle(2, [(-2, 3), (-2, 3)])
#
# f_grad = lambda x: np.array([2 * (x[0] - 2), 2 * x[1] + 2])
#
# real_solution = np.array([2, -1])
# x0 = np.array([-2,2])
# x1 = np.array([-1,1])
# def_lam = 0.1
#
# problem = FuncNDMin(2, f_to_min, f_grad, C=C, x0=x0, hr_name='$f(x) -> min, C = {0}$'.format(C), xtest=real_solution)

# endregion

# region N-d func min with R+ projection - (x-x_1)^2 + (x-x_2)^2 + ... + (x-x_n)^2
# N: int = 100
#
# real_solution: np.ndarray = np.random.randint(-3, 3, N)
#
# f_to_min = lambda x: np.sum([(x[i] - real_solution[i]) ** 2 for i in range(N)])
# f_grad = lambda x: np.array([2 * (x[i] - real_solution[i]) for i in range(N)])
#
# # cut to R+
# C = Hyperrectangle(N, [
#     [0, real_solution[i]+(np.random.random())] for i in range(N)
# ])
#
# real_solution = np.array([(0 if s < 0 else s) for s in real_solution])
#
# x0 = np.random.randn(N)
# x1 = np.random.randn(N)
#
# def_lam = 0.01
#
# problem = FuncNDMin(N, f_to_min, f_grad, C=C, x0=x0, hr_name='$f(x) -> min, C = {0}$'.format(C), xtest=real_solution)
# endregion

# region N-d func min with Hypercube and Hyperplane projection - (x-x_1)^2 + (x-x_2)^2 + ... + (x-x_n)^2
# N: int = 10
#
# real_solution: np.ndarray = np.random.randint(-3, 3, N)
#
# f_to_min = lambda x: np.sum([(x[i] - real_solution[i]) ** 2 for i in range(N)])
# f_grad = lambda x: np.array([2 * (x[i] - real_solution[i]) for i in range(N)])
#
# bounds_temp = [[np.random.random()*2 - 2., 0] for i in range(N)]
# bounds = [[b[0], b[0] + np.random.randint(1,3)] for b in bounds_temp]
#
# C1 = Hyperrectangle(N, bounds)
# C2 = Hyperplane(a=np.array([1 for i in range(N)]), b=1)
#
# C = ConvexSetsIntersection([C1,C2])
#
# print(f"Real solution before projection: {real_solution}. Distance to C: {C.getDistance(real_solution)}")
#
# real_solution_proj = C.project(real_solution)
#
# if C1.isIn(real_solution_proj) and C2.isIn(real_solution_proj):
#     print(f"Projected solution tested. Distance to C: {C.getDistance(real_solution_proj)}")
#     print(f"{real_solution} -> {real_solution_proj}")
# else:
#     print(f"!!! Projected solution is outside C!!! Distance to C: {C.getDistance(real_solution_proj)}")
#     print(real_solution_proj)
#
# x0 = np.random.randn(N)
# x1 = np.random.randn(N)
#
# def_lam = 0.01
#
# problem = FuncNDMin(N, f_to_min, f_grad, C=C, x0=x0, hr_name='$f(x) -> min, C = {0}$'.format(C), xtest=real_solution_proj)
# endregion

# region (X1+X2+...+Xn - n/2)^2 -> min; lam = 1/4N; multiple solutions

# N = 10
# def_lam = 1./(4.*N)
#
# x0 = np.array([i+1 for i in range(N)])
# x1 = np.array([i+0.5 for i in range(N)])
#
# real_solution = np.array([0.5 for i in range(N)])
#
# hr = Hyperrectangle(N, [[-1,1] for i in range(N)])
# constraints = hr
#
# #hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
# #hp = Hyperplane(a=np.array([1., 1., 1.]), b=3./2.)
# # constraints = ConvexSetsIntersection([hr, hp])
#
# problem = FuncNDMin(N,
#               lambda x: (np.sum(x) - N/2) ** 2,
#               lambda x: np.ones(N) * 2 * (np.sum(x) - N/2),
#               C=constraints,
#               x0=x0,
#               hr_name='$(x_1 + x_2 + ... + x_n - n/2)^2->min, C = [-5,5]x[-5,5], N = {0}$'.format(N),
#               xtest=real_solution
#               )

# endregion


# region Test problem initialization

# problem = pseudo_mono_3.prepareProblem(algorithm_params=params)
# problem = pseudo_mono_5.prepareProblem(algorithm_params=params)

# problem = harker_test.prepareProblem(algorithm_params=params)
problem = minmax_game_1.prepareProblem(algorithm_params=params)

# problem = sle_saddle_regression_100_100000.prepareProblem(algorithm_params=params)

# problem = sle_saddle_hardcoded.prepareProblem(algorithm_params=params)
# problem = sle_saddle_random_one.prepareProblem(algorithm_params=params)

# problem = pigu_sample.prepareProblem(algorithm_params=params)
# problem = braess_sample.prepareProblem(algorithm_params=params)

# problem = load_file_sample.prepareProblem(algorithm_params=params)
# sys.stdout = sys.__stdout__
# print(captured_io.getvalue())
# sys.stdout = captured_io
# exit(0)
# endregion


# region SLAE with HR and HP projection
# def_lam = 0.005
# def_adapt_lam = 0.5
#
# isLoad = True
# matrixProblemId = 20211
# baseProblemPath='storage/data/BadMatrix100-1/'
#
# if not isLoad:
#     N = 10
#     maxEl = 5
#     isOk = False
#
#     while not isOk:
#         A = np.random.rand(N, N)*maxEl
#
#         A = A.T*A
#         A += (N*maxEl/5)*np.identity(N)
#         testX = np.ones(N, dtype=float)
#
#         isOk = np.all(np.linalg.eigvals(A) > 0)
#
#     if isOk:
#         x0 = testX + (np.random.rand(N)*10 - 5)
#         x1 = testX + (np.random.rand(N) * 20 - 10)
#         np.save(baseProblemPath+'A'+str(matrixProblemId)+'.data', A)
#         np.save(baseProblemPath+'x0'+str(matrixProblemId)+'.data', x0)
#     else:
#         print("Error: matric is not PD!")
#         exit()
# else:
#     A = np.load(baseProblemPath+'A'+str(matrixProblemId)+'.data.npy')
#     x0 = np.load(baseProblemPath+'x0'+str(matrixProblemId)+'.data.npy')
#
#     N = x0.shape[0]
#     x1 = x0 + (np.random.rand(N) * 2 - 1)
#
#     testX = np.ones(N, dtype=float)
#
# C1 = Hyperrectangle(N, [(-0.5, 0.5) for i in range(N)])
# C2 = Hyperplane(a=np.array([1. for i in range(N)]), b = 1.0)
# C = ConvexSetsIntersection([C1, C2])
#
# projected_test = C.project(testX)
#
# problem = MatrixOperVI(A=A, b=A @ testX, x0=x0, C = C,
#              hr_name='$N='+str(N)+'$',
#              xtest=projected_test)
# endregion

# region SLE direct on L1 ball - 3x3, predefined
# N = 3
#
# M, p, unconstrained_solution = getSLE(N, A=np.array([
#           [5, 2,  1]
#         , [2, 13, 4]
#         , [1, 4, 6]
#     ], dtype=float), x_test=np.array([1, 2, 3]))
#
# norm = np.linalg.norm(M, 2)
#
# c = np.sum(np.abs(unconstrained_solution)) * 1.9
# c = 3.
#
# x0 = np.array([0.2 for i in range(N)])
# x1 = np.array([0.1 for i in range(N)])
# # def_lam = 1./norm
# def_lam = 0.0005
# def_adapt_lam = 0.1
# constraints = L1Ball(N, c)
#
# projected_solution = constraints.project(unconstrained_solution)
# print("M:")
# print(M)
# print(f"P: {p}")
# print(f"Projected solution: {projected_solution}; c: {c}")
# print(f"Goal F on proj. sol.: {np.linalg.norm(M @ projected_solution - p)}")
#
# # cvx_begin
# #     variable x(n);
# #     minimize( norm(A*x-b) );
# #     subject to
# #         C*x == d;
# #         norm(x,Inf) <= 1;
# # cvx_end
#
# x = cp.Variable(N)
# objective = cp.Minimize(cp.sum_squares(M@x - p))
# constraints_cp = [cp.norm(x, 1) <= c]
# print(f"constraints_cp: {constraints_cp[0].is_dcp()}, ")
#
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
#
# problem = SLEDirect(
#               M=M, p=p,
#               C=constraints,
#               x0=x0,
#               x_test=test_solution,
#               hr_name='$||Mx - p||_2 \\to min, ||x|| <= '+ str(c) +' \ \lambda = ' + str(round(def_lam, 3)) + '$'
#               )
#
# endregion

# region SLE direct on L1 ball - 2x3, predefined
# n = 2
# m = 3
#
# M = np.array([
#           [5, 2,  1]
#         , [2, 13, 4]
#     ], dtype=float)
#
# norm = np.linalg.norm(M, 2)
#
# unconstrained_solution = np.array([-1, 1, 0])
#
# p = M @ unconstrained_solution
# c = 1.
#
# x0 = np.array([0.2 for i in range(m)])
# x1 = np.array([0.1 for i in range(m)])
# # def_lam = 1./norm
# def_lam = 0.0001
# def_adapt_lam1 = 0.2
# constraints = L1Ball(m, c)
#
# projected_solution = constraints.project(unconstrained_solution)
# print("M:")
# print(M)
# print(f"P: {p}")
# print(f"Projected solution: {projected_solution}; c: {c}")
# print(f"Goal F on proj. sol.: {np.linalg.norm(M @ projected_solution - p)}")
#
# x = cp.Variable(m)
# objective = cp.Minimize(cp.sum_squares(M@x - p))
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
#
# problem = SLEDirect(
#               M=M, p=p,
#               C=constraints,
#               x0=x0,
#               x_test=test_solution,
#               hr_name='$||Mx - p||_2 \\to min, ||x|| <= '+ str(c) +' \ \lambda = ' + str(round(def_lam, 3)) + '$'
#               )
#
# endregion

# region SLE direct on L1 ball - nxm, random
# n = 10
# m = 20
#
# M = np.random.rand(n, m) * 3.
#
# norm = np.linalg.norm(M, 2)
#
# unconstrained_solution = np.array([i*0.1 for i in range(m)])
#
# p = M @ unconstrained_solution
# c = 2.
#
# x0 = np.array([0.2 for i in range(m)])
# x1 = np.array([0.1 for i in range(m)])
# #def_lam = 1./norm
# def_lam = 0.001
# def_adapt_lam1 = 0.01
# constraints = L1Ball(m, c)
#
# projected_solution = constraints.project(unconstrained_solution)
# print("M:")
# print(M)
# print(f"P: {p}")
# print(f"Projected solution: {projected_solution}; c: {c}")
# print(f"Goal F on proj. sol.: {np.linalg.norm(M @ projected_solution - p)}")
#
# x = cp.Variable(m)
# objective = cp.Minimize(cp.sum_squares(M@x - p))
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
# print(f"CP solution distance to C: {constraints.getDistance(test_solution)}")
#
# problem = SLEDirect(
#               M=M, p=p,
#               C=constraints,
#               x0=x0,
#               x_test=test_solution,
#               hr_name='$rand ||Mx - p||_2 \\to min, ||x|| <= '+ str(c) +' \ \lambda = ' + str(round(def_lam, 3)) + '$'
#               )

# endregion

# region SLE saddle form on L1 ball - nxm, random
# n = 5
# m = 10
#
# M = np.random.rand(n, m) * 3.
#
# norm = np.linalg.norm(M, 2)
#
# unconstrained_solution = np.array([1. for i in range(m)])
#
# p = M @ unconstrained_solution
#
# c = 1.5
# # c = 1.34
#
# x0 = np.array([0.2 for i in range(m)])
# x1 = np.array([0.1 for i in range(m + n)])
# # def_lam = 1./norm
# def_lam = 0.01
# def_adapt_lam1 = 0.5
# max_iters = 2000
# constraints = L1Ball(m, c)
#
# projected_solution = constraints.project(unconstrained_solution)
# print("M:")
# print(M)
# print(f"P: {p}")
# print(f"Projected solution: {projected_solution}; c: {c}")
# print(f"Goal F on proj. sol.: {np.linalg.norm(M @ projected_solution - p)}")
# print()
#
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
#
# problem = SLESaddle(
#     M=M, p=p,
#     C=constraints,
#     x0=x0,
#     x_test=test_solution,
#     hr_name='$||Mx - p||_2 \\to min, minimax \ form, ||x||_1 \leq ' + str(c) + ' \ \lambda = ' + str(round(def_lam, 3)) + '$'
# )

# endregion

# region Init all algs

def initAlgs():
    korpele = Korpelevich(problem, eps=params.eps, lam=params.lam, min_iters=params.min_iters, max_iters=params.max_iters)
    korpele_adapt = KorpelevichMod(problem, eps=params.eps, min_iters=params.min_iters, max_iters=params.max_iters)

    tseng_adaptive = TsengAdaptive(problem,
                                   eps=params.eps, lam=params.start_adaptive_lam, tau=params.adaptive_tau,
                                   min_iters=params.min_iters, max_iters=params.max_iters, hr_name="Alg. 4 (tseng-A)")

    tseng = Tseng(problem,
                  eps=params.eps, lam=params.lam,
                  min_iters=params.min_iters, max_iters=params.max_iters, hr_name="Tseng")

    tseng_bregproj = Tseng(problem,
                           eps=params.eps, lam=params.lam_KL,
                           min_iters=params.min_iters, max_iters=params.max_iters,
                           hr_name="Tseng-BP", projection_type=ProjectionType.BREGMAN)

    extrapol_from_past = ExtrapolationFromPast(problem, stop_condition=params.stop_by,
                                               y0=params.x1.copy(), eps=params.eps, lam=params.lam*(math.sqrt(2.)-1),
                                               min_iters=params.min_iters, max_iters=params.max_iters, hr_name="Alg. 1 - E (S)")

    extrapol_from_past_bregproj = ExtrapolationFromPast(problem, stop_condition=params.stop_by,
                                                        y0=params.x1.copy(), eps=params.eps, lam=params.lam_KL * (math.sqrt(2.) - 1),
                                                        min_iters=params.min_iters, max_iters=params.max_iters,
                                                        hr_name="Alg. 1 - KL (S)", projection_type=ProjectionType.BREGMAN)

    extrapol_from_past_adaptive = ExtrapolationFromPastAdapt(problem, stop_condition=params.stop_by,
                                                             y0=params.x1.copy(), eps=params.eps,
                                                             lam=params.start_adaptive_lam1, tau=params.adaptive_tau_small,
                                                             min_iters=params.min_iters, max_iters=params.max_iters,
                                                             hr_name="Alg. 1 - E")

    extrapol_from_past_adaptive_bregproj = ExtrapolationFromPastAdapt(problem, stop_condition=params.stop_by,
                                                             y0=params.x1.copy(), eps=params.eps,
                                                             lam=params.start_adaptive_lam1, tau=params.adaptive_tau_small,
                                                             min_iters=params.min_iters, max_iters=params.max_iters,
                                                             hr_name="Alg. 1 - KL", projection_type=ProjectionType.BREGMAN)


    malitsky_tam = MalitskyTam(problem, stop_condition=params.stop_by,
                               x1=params.x1.copy(), eps=params.eps, lam=params.lam/2.,
                               min_iters=params.min_iters, max_iters=params.max_iters, hr_name="Alg. 2 - E (S)")

    malitsky_tam_bregproj = MalitskyTam(problem, stop_condition=params.stop_by,
                                        x1=params.x1.copy(), eps=params.eps, lam=params.lam_KL / 2.,
                                        min_iters=params.min_iters, max_iters=params.max_iters,
                                        hr_name="Alg. 2 - KL (S)", projection_type=ProjectionType.BREGMAN)

    malitsky_tam_adaptive = MalitskyTamAdaptive(problem,
                                                x1=params.x1.copy(), eps=params.eps, stop_condition=params.stop_by,
                                                lam=params.start_adaptive_lam1, lam1=params.start_adaptive_lam1,
                                                tau=params.adaptive_tau,
                                                min_iters=params.min_iters, max_iters=params.max_iters, hr_name="Alg. 2 - E")

    malitsky_tam_adaptive_bregproj = MalitskyTamAdaptive(problem,
                                                x1=params.x1.copy(), eps=params.eps, stop_condition=params.stop_by,
                                                lam=params.start_adaptive_lam1, lam1=params.start_adaptive_lam1,
                                                tau=params.adaptive_tau,
                                                min_iters=params.min_iters, max_iters=params.max_iters,
                                                hr_name="Alg. 2 - KL", projection_type=ProjectionType.BREGMAN)



    algs_to_test = [
        # korpele,
        # korpele_adapt,
        # malitsky_tam_adaptive,
    #    tseng,
    #    tseng_bregproj,
#        tseng_adaptive,
        extrapol_from_past,
        extrapol_from_past_bregproj,
        extrapol_from_past_adaptive,
        extrapol_from_past_adaptive_bregproj,
        malitsky_tam,
        malitsky_tam_bregproj,
        malitsky_tam_adaptive,
        malitsky_tam_adaptive_bregproj
    ]

    return algs_to_test

algs_to_test = initAlgs()
# endregion

# region Run all algs and save data and results
saved_history_dir = "storage/stats2021-11-traffic"
test_mnemo = f"{problem.__class__.__name__}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
saved_history_dir = os.path.join(saved_history_dir, test_mnemo)
os.makedirs(saved_history_dir, exist_ok=True)

problem.saveToDir(path_to_save=os.path.join(saved_history_dir, "problem"))
params.saveToDir(os.path.join(saved_history_dir, "params"))
print(f"eps: {params.eps}; tau1: {params.adaptive_tau}; tau2: {params.adaptive_tau_small}; "
      f"start_lam: {params.start_adaptive_lam}; start_lam1: {params.start_adaptive_lam1}; "
      f"lam: {params.lam}; lam_KL: {params.lam_KL}")

if params.save_history:
    writer = pd.ExcelWriter(
        os.path.join(saved_history_dir, f"history-{test_mnemo}.xlsx"),
        engine='openpyxl')

alg_history_list = []
for alg in algs_to_test:
    if params.test_time:
        total_time: int = 0
        for i in range(params.test_time_count):
            # alg.history = AlgHistory(alg.history.x.shape[1])
            alg.do()
            if alg.isStopConditionMet() and alg.iter < params.max_iters:
                total_time += alg.history.iter_time_ns[alg.history.iters_count - 1]
            else:
                total_time = math.inf
            print(f"{i+1}: {alg.hr_name} time: {alg.history.iter_time_ns[alg.history.iters_count - 1]/params.time_scale_divider}s.")

        BasicAlgoTests.PrintAlgRunStats(alg)
        # print(f"{alg.hr_name} average time from {params.test_time_count} repeats: {float(total_time)/params.test_time_count/params.time_scale_divider}s.")

    else:
        alg.do()
        BasicAlgoTests.PrintAlgRunStats(alg)
        alg_history_list.append(alg.history)

        if params.save_history:
            df = alg.history.toPandasDF()
            df.to_excel(writer, sheet_name=alg.hr_name.replace("*", "_star"), index=False)
        print('')

        np.save('traff_eq_lastx', alg.history.x[alg.history.iters_count - 1])

if params.save_history:
    writer.save()
    writer.close()

sys.stdout = sys.__stdout__
print(captured_io.getvalue())

f = open(os.path.join(saved_history_dir, f"log-{test_mnemo}.txt"), "w")
f.write(captured_io.getvalue())
f.close()

# save history - takes too much space for big matrices!
# for idx, h in enumerate(alg_history_list):
#     hp = os.path.join(saved_history_dir, 'history', str(idx))
#     h.saveToDir(hp)

# endregion


# region Plot and save graphs
if params.show_plots:
    grapher = AlgStatGrapher()
    grapher.plot_by_history(
        alg_history_list=alg_history_list,
        x_axis_type=params.x_axis_type, y_axis_type=params.y_axis_type, y_axis_label=params.y_label,
        styles=params.styles, start_iter=params.plot_start_iter,
        x_axis_label=params.x_label, time_scale_divider=params.time_scale_divider
    )

    if params.x_limits is not None:
        plt.ylim(params.x_limits)

    if params.y_limits is not None:
        plt.ylim(params.y_limits)

    dpi = 300.

    plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.svg"), bbox_inches='tight', dpi=dpi, format='svg')
    plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.eps"), bbox_inches='tight', dpi=dpi, format='eps')

    plt.title(problem.hr_name, loc='center')
    plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.png"), bbox_inches='tight', dpi=dpi)

    plt.show()
    exit()

# endregion



# table - time for getting to epsilon error
# for 1 and 2 - "real error"
# for 3 - step error (can be multiple solutions) and see F(x)
# show lambda on separate graph
