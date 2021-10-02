import math

import numpy as np
from matplotlib import pyplot as plt

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.classic_simplex import ClassicSimplex
from constraints.hyperplane import Hyperplane
from methods.korpele_mod import KorpelevichMod
from methods.malitsky_tam import MalitskyTam
from methods.malitsky_tam_adaptive import MalitskyTamAdaptive
from methods.tseng import Tseng
from methods.tseng_adaptive import TsengAdaptive
from problems.harker_test import HarkerTest
from problems.matrix_oper_vi import MatrixOperVI
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from utils.graph.alg_stat_grapher import AlgStatGrapher

from constraints.hyperrectangle import Hyperrectangle

from methods.korpelevich import Korpelevich
from problems.funcndmin import FuncNDMin
from utils.test_alghos import BasicAlgoTests

min_iters = 10
max_iters = 2000
def_lam = 0.005
def_adapt_lam = 0.1
def_eps = 1e-8

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

# region PseudoMonotone One
N = 3

x0 = np.array([2., -5., 3.])
x1 = np.array([3., -2., -1.])
def_lam = 0.05
def_adapt_lam = 1.

real_solution = np.array([0.0 for i in range(N)])

hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

constraints = ConvexSetsIntersection([hr, hp])

problem = PseudoMonotoneOperOne(
              C=constraints,
              x0=x0,
              hr_name='$(x_1 + x_2 + +x_3 = 0, C = [-5,5]^3, N = 3$'
              )
# endregion

# region HarkerTest
# N = 5
#
# x0 = np.array([0.5 for i in range(N)])
# x1 = np.array([0.25 for i in range(N)])
#
# real_solution = np.array([0.0 for i in range(N)])
#
# #ht = HarkerTest(N, C=ClassicSimplex(N, N), hr_name='HarkerTest', x0=x0, xtest=real_solution)
# ht = HarkerTest(N, C=Hyperrectangle(N, [(-5,5) for i in range(N)]), hr_name='HarkerTest', x0=x0, xtest=real_solution)
#
# problem = ht
# def_lam = 0.4/ht.norm
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


korpele = Korpelevich(problem, eps=def_eps, lam=def_lam, min_iters=min_iters, max_iters=max_iters)
korpele_adapt = KorpelevichMod(problem, eps=def_eps, min_iters=min_iters, max_iters=max_iters)

tseng = Tseng(problem, eps=def_eps, lam=def_lam, min_iters=min_iters, max_iters=max_iters)
tseng_adaptive = TsengAdaptive(problem, eps=def_eps, lam=def_adapt_lam, min_iters=min_iters, max_iters=max_iters)

malitsky_tam = MalitskyTam(problem, x1=x1.copy(), eps=def_eps, lam=def_lam, min_iters=min_iters, max_iters=max_iters)
malitsky_tam_adaptive = MalitskyTamAdaptive(problem, x1=x1.copy(), eps=def_eps, lam=def_adapt_lam,
                                            min_iters=min_iters, max_iters=max_iters)

algs_to_test = [
    korpele,
    korpele_adapt,
    tseng,
    tseng_adaptive,
    malitsky_tam,
    malitsky_tam_adaptive
]

alg_history_list = []
for alg in algs_to_test:
    alg.do()
    BasicAlgoTests.PrintAlgRunStats(alg)
    alg_history_list.append(alg.history)
    print('')

grapher = AlgStatGrapher()
grapher.plot_by_history(
    alg_history_list=alg_history_list,
    plot_step_delta=False, plot_real_error=True,
    x_axis_label='Iterations', y_axis_label='Real error', plot_title=problem.hr_name
)

plt.show()

exit()

# table - time for getting to epsilon error
# for 1 and 2 - "real error"
# for 3 - step error (can be multiple solutions) and see F(x)
# show lambda on separate graph
