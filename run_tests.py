import time
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List

from constraints.classic_simplex import ClassicSimplex
from constraints.allspace import Rn
from methods.batched_grad_proj import BatchedGradProj
from methods.korpele_vari_x_y import KorpeleVariX_Y
from methods.korpele_mod import KorpelevichMod
from methods.semenov_forback import SemenovForBack
from methods.varistepthree import VaristepThree
from methods.varisteptwo import VaristepTwo
from problems.func_sum_min_simple import FuncSumMinSimple
from problems.harker_test import HarkerTest
from problems.koshima_shindo import KoshimaShindo
from problems.lin_sys_splitted import LinSysSplitted
from problems.lin_sys_splitted_l1 import LinSysSplittedL1
from problems.log_reg_flavor_one import LogRegFlavorOne
from problems.log_reg_flavor_two import LogRegFlavorTwo
from problems.linear_prog import LinearProgProblem
from problems.func_saddle_point import FuncSaddlePoint

from problems.matrix_oper_vi import MatrixOperVI
from problems.nonlin_r2_oper import NonlinR2Oper
from problems.page_rank_problem import PageRankProblem
from problems.problem import Problem
from problems.visual_params import VisualParams
from utils.graph.alg_stat_grapher import AlgStatGrapher

from methods.grad_proj import GradProj
from methods.korpelevich import Korpelevich
from methods.varistepone import VaristepOne
from methods.popov_subgrad import PopovSubgrad

from constraints.hyperrectangle import Hyperrectangle
from constraints.positive_simplex_area import PositiveSimplexArea
from problems.funcndmin import FuncNDMin
from utils.test_alghos import BasicAlghoTests
from problems.testcases.matrix_grad_fail import getProblem

from utils.print_utils import vectorToString


# T = HarkerTest(4)
# exit(1)

# region common functions - alg state event listeners, savers etc
def saveStats(alg, name, stat, currentState):
    basePath = "{0}/{1:%y_%m_%d_%H_%M_%S}/{2}".format(dataPath, tryTimestamp, name)
    os.makedirs(basePath, exist_ok=True)
    np.savetxt("{0}/{1}".format(basePath, 'stat.txt'), stat)
    np.savetxt("{0}/{1}".format(basePath, 'res.txt'), currentState['x'][0])


def onAlgStart(alg, currentState):
    global stat
    print("Alg {0} started.".format(type(alg).__name__))
    stat.append([])


def onAlgFinish(alg, currentState):
    global stat
    global statIdx
    saveStats(alg, type(alg).__name__, stat[statIdx], currentState)
    statIdx += 1

    # time.sleep(1)


def onIter(alg, currentState, iterNum, timeElapsed):
    # print("{0}: {1}; D: {2:.4f}; R: {3:.4f}".format(iterNum, timeElapsed, currentState['D'], np.dot(currentState['x'][2],currentState['x'][2])))
    if iterNum > 1:
        stat[statIdx].append([
            iterNum, #0
            timeElapsed, #1
            #2
            currentState['D'][0] if isinstance(currentState['D'], tuple) else currentState['D'],
            #3
            currentState['D'][1] if isinstance(currentState['D'], tuple) and len(currentState['D'])>1 else
            currentState['D'][0] if isinstance(currentState['D'], tuple) else currentState['D'],
            #4
            currentState['F'][0],
            #5
            currentState['F'][1] if len(currentState['F'])>1 else currentState['F'][0]]
        )

    return True


def allAlgsFinished():
    global stat
    global statIdx
    print('All tests finished.')


# endregion

# region default parameters
do_graph = True

lam = 0.1
lamInit = 1.0
eps = 1e-8
tet = 0.9
tau = 0.5
sigma = 1.0
stab = 0
N = 2

minIterTime = 0
printIterEvery = 100
maxIters = 10000
minIters = 0

extraTitle = ''

dataPath = 'storage/methodstats'
# endregion

# region problems
problems: List[Problem] = []

# 4d funcmin
# problems.append(
#     FuncNDMin(3,
#               lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2,
#               lambda x: np.array([2 * x[0], 2 * x[1], 2 * x[2]]),
#               C=Hyperrectangle(3, [(-10, 10), (-10, 10), (-10, 10)]),
#               x0=np.array([2, 2, -7]),
#               xtest=np.array([1, 0, 0]),
#               L=10,
#               vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#               hr_name='$x^2+y^2+z^2->min, x \in R^3$'
#               )
# )
#

# problems.append(
#     FuncNDMin(3,
#               lambda x: (x[0] - 1) ** 2 + x[1] ** 2 + (x[2] + 2) ** 2,
#               lambda x: np.array([2 * x[0] - 2, 2 * x[1], 2 * (x[2] + 2)]),
#               C=Hyperrectangle(3, [(-10, 10), (-10.3, 13), (-10.5, 13)]),
#               x0=np.array([2, 2, -7]),
#               xtest=np.array([1, 0, -2]),
#               L=10,
#               vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#               hr_name='$(x-1)^2+y^2+(z+2)^2->min, C = [-10,0.6]x[0.3,13]x[-1.5,13]$'
#               )
# )

# N = 3
# problems.append(
#     FuncSumMinSimple(3,
#               [lambda x: (x[0] - 1) ** 2, lambda x: x[1] ** 2, lambda x: (x[2] + 2) ** 2],
#               [
#                          lambda x: np.array([2 * x[0] - 2, 0, 0]),
#                          lambda x: np.array([0, 2 * x[1], 0]),
#                          lambda x: np.array([0, 0, 2 * (x[2] + 2)])
#                ],
#               C=Hyperrectangle(3, [(-10, 10), (-10.3, 13), (-10.5, 13)]),
#               x0=np.array([2, 2, -7]),
#               xtest=np.array([1, 0, -2]),
#               vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#               hr_name='$(x-1)^2+y^2+(z+2)^2->min, C = [-10,0.6]x[0.3,13]x[-1.5,13]$'
#               )
# )

# X = np.array([
#     [1, 1, 1],
#     [-1, 1, 1],
#     [1, 1, -1],
#     [-31, 1, -1],
#     [31, 1, 1],
#     [-12, 1, 1],
#     [21, 1, -1],
#     [-1, 45, 1],
#     [-1, 4, 1],
#     [-1, -31, 1],
#     [1, 33, -1]
# ])
#
# y = np.array([np.sign(t[-1]) for t in X])
#
# print("LogRegCont: X: {0}; y:{1} ", (np.dot(A@x0 - b, A@x0 - b)))
# print("Initial error 1: ", (np.abs(A@x0 - b).sum()))
#
# wtest = np.array([0, 0, 1])
# w0 = np.array([1., 1, 1.0])

# problems.append(
#     LogRegFlavorOne(X, y,
#                     C=Hyperrectangle(X.shape[1], [(-100, 100) for i in range(X.shape[1])]),
#                     w0=w0,
#                     wtest=wtest,
#                     hr_name='LogResCont',
#                     lam_override=0.1
#                     )
# )


# region SLAE problem #2

# isLoad = True
# matrixProblemId = 3
# baseProblemPath='storage/data/BadMatrix100-1/'
#
# if not isLoad:
#     N = 100
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
#         np.save(baseProblemPath+'A'+str(matrixProblemId)+'.data', A)
#         np.save(baseProblemPath+'x0'+str(matrixProblemId)+'.data', x0)
#     else:
#         print("Error: matric is not PD!")
# else:
#     A = np.load(baseProblemPath+'A'+str(matrixProblemId)+'.data.npy')
#     x0 = np.load(baseProblemPath+'x0'+str(matrixProblemId)+'.data.npy')
#
#     N = x0.shape[0]
#
#     testX = np.ones(N, dtype=float)
#
# C=Hyperrectangle(N, [(-5, 5) for i in range(N)])
#
# lam_override = 0.0001385
#
# problems.append(
#     MatrixOperVI(A=A, b=A @ testX, x0=x0, C = C,
#                  hr_name='$Ax=b; N='+str(N)+';\lambda='+str(lam_override)+'$', xtest=testX, lam_override=lam_override)
# )

# endregion

# region (X1+X2+...+Xn - n/2)^2 -> min; lam = 1/4N

# N = 100
# problems.append(
#     FuncNDMin(N,
#               lambda x: (np.sum(x) - N/2) ** 2,
#               lambda x: np.ones(N) * 2 * (np.sum(x) - N/2),
#               C=Hyperrectangle(N, [(0, 5) for i in range(N)]),
#               x0=np.array([i+1 for i in range(N)]),
#               hr_name='$(x + y -1)^2->min, C = [-5,5]x[-5,5]$',
#               lam_override=1.0/N/4
#               )
# )

# endregion

# N = 100
# A = np.identity(N, dtype=float)
# # #A = np.random.rand(dim, dim)
# A[0,3] = 1
# A[3,0] = 1
# A[N-1,7] = -1
# A[7,N-1] = -1
# testX = np.ones(N, dtype=float)
# x0 = np.random.rand(N)
#
# hr_bounds = [(-5,5) for i in range(N)]
#
# problems.append(
#     MatrixOperVI(A=A, b=A @ testX, x0=x0,
#                  hr_name='$Ax=b$')
# )

# problems.append(
#      NonlinR2Oper(x0=np.array([1,1]), hr_name='$NonLinA$')
# )

# N = 20
# hr_bounds = [(-5,5) for i in range(N)]
# problems.append(HarkerTest(N, C=Hyperrectangle(N, hr_bounds), hr_name='HPHard'),)

# ht = HarkerTest(N, C=PositiveSimplexArea(N, 4), hr_name='HarkerTest', x0=np.ones(N))
# problems.append(ht,)
# lam = 0.4/ht.norm
# print('Lam: ', lam)

# rotP = getProblem(N, False, False)
# rotP.hr_name = 'Приклад 1'
# problems.append(rotP)

# problems.append(KoshimaShindo(x0=np.random.rand(4)))

# N = 5
# ppr = PageRankProblem.CreateRandom(N, 0.01)
# problems.append(ppr)

# region LinearProgProblems
# N = 5
# p = LinearProgProblem(
#     A=np.array([
#         [-1, 0]
#         ,[-1, -1]
#         ,[0, -1]
#         ,[1, 0]
#         ,[-0.25, 1]
#     ]),
#     b=np.array([-1, -4, 0, 9, 23.0*0.25]),
#     c=np.array([1, -1]),
#     x0=np.array([4, 4, 1, 1, 1, 1, 1])
# )
#
# problems.append(p)
# endregion

# region plain Saddle point problems

# N = 2
# # x^2 - y^2
# problems.append(
#     FuncSaddlePoint(arity=N, f=lambda x: x[0] ** 2 - x[1] ** 2,
#                     dfConvex=lambda x: np.array([x[0] * 2]), dfConcave=lambda x: np.array([-x[1] * 2]),
#                     convexVarIndices=[0], concaveVarIndices=[1],
#                     C=Rn(N),
#                     x0=np.array([2, 3]),
#                     xtest=np.array([0, 0]),
#                     L=10,
#                     vis=[VisualParams(xl=-5, xr=5, yb=-5, yt=5, zn=0, zf=56, elev=22, azim=-49)],
#                     hr_name='$x^2 - y^2->SP, (x,y) \in R^2$'
#                     )
# )

# N = 3
# # (x-1.3)^2 - (y-2.1)^2 + z^2
# problems.append(
#     FuncSaddlePoint(arity=N, f=lambda x: (x[0]-1.3) ** 2 - (x[1]-2.1) ** 2 + x[2] ** 2,
#                     gradF=lambda x: np.array([(x[0]-1.3) * 2, -(x[1] - 2.1) * 2, x[2] * 2]),
#                     convexVarIndices=[0,2], concaveVarIndices=[1],
#                     C=Rn(N),
#                     x0=np.array([1,1,1]),
#                     xtest=np.array([0, 0, 0]),
#                     L=10,
#                     vis=[VisualParams(xl=-5, xr=5, yb=-5, yt=5, zn=0, zf=56, elev=22, azim=-49)],
#                     hr_name='$x^2 - y^2 + z^2->SP, (x,y, z) \in R^3$'
#                     )
# )

# endregion

# endregion

# region tests and plots
for p in problems:

    grad_desc = GradProj(p, eps, lam, min_iters=minIters)
    popov_subgrad = PopovSubgrad(p, eps, lam, min_iters=minIters)
    popov_subgrad.hr_name = 'Popov'
    korpele_basic = Korpelevich(p, eps, p.GetLambdaOverride() if p.GetLambdaOverride() else lam, min_iters=minIters)
    korpele_basic.hr_name = 'Korpelevich'
    korpele_mod = KorpelevichMod(p, eps, lam, min_iters=minIters)
    semenov_forback = SemenovForBack(p, eps, 0.001, min_iters=minIters)

    varistepone = VaristepOne(p, eps, lam, min_iters=minIters)
    varistepone.hr_name = 'Alg1'
    korpele_vari_x_y = KorpeleVariX_Y(p, eps, lamInit, min_iters=minIters, phi=0.75, gap=-1)
    extraTitle = '$;\phi='+str(korpele_vari_x_y.phi) + (
                 ';$ no $\lambda$ increase' if korpele_vari_x_y.gap < 0 else '; \lambda$ inc every '+str(korpele_vari_x_y.gap+1) + ' iters')

    # varisteptwo = VaristepTwo(p, eps, lam, min_iters=minIters, xstar=np.array([1.2247, 0, 0, 2.7753]))
    # varistepthree = VaristepThree(p, eps, lam, min_iters=minIters, xstar=np.array([1.2247, 0, 0, 2.7753]))
    xst = np.ones(N)
    # xst[0]=0

    varisteptwo = VaristepTwo(p, eps, lam, min_iters=minIters, xstar=xst, alfa_calc=lambda i: 1.0 / (i + 1))
    varisteptwo.hr_name = 'Alg2'
    varistepthree = VaristepThree(p, eps, lam, min_iters=minIters)
    varistepthree.hr_name = 'Alg3'

    tryTimestamp = datetime.now()
    statIdx = 0
    stat = []

    tested_items = [
        #grad_desc,
        #,
        #,varistepone
        # ,varisteptwo
        # ,varistepthree
        #,
        #korpele_vari_x_y
      korpele_basic
        #semenov_forback
        #,korpele_mod
        # ,popov_subgrad
    ]

    alghoTester = BasicAlghoTests(print_every=printIterEvery, max_iters=maxIters, min_time=minIterTime,
                                  on_alg_start=onAlgStart, on_alg_finish=onAlgFinish, on_iteration=onIter)

    res = alghoTester.DoTests(tested_items)

    if do_graph:
        grapher = AlgStatGrapher()

        if statIdx > 0:
            statsAsArray = np.array(stat)
            grapher.plot(statsAsArray, xDataIndices=[1 for i in range(len(tested_items))],
                         yDataIndices=[[2, 4] for i in range(len(tested_items))],
                         plotTitle=p.GetHRName() + (' ' + extraTitle if extraTitle != '' else ''),
                         xLabel='Час, c.', yLabel='$||x_{n}-y_n||^2, ||F(x)||^2$',
                         legend=[[it.GetHRName() + ' $||x_n-y_n||^2$', it.GetHRName() + " $||F(x)||^2$"] for it in tested_items])

            grapher.plot(statsAsArray, xDataIndices=[0 for i in range(len(tested_items))],
                         yDataIndices=[[2, 4] for i in range(len(tested_items))],
                         plotTitle=p.GetHRName() + (' ' + extraTitle if extraTitle != '' else ''),
                         xLabel='Кількість ітерацій $n$', yLabel='$||x_{n}-y_n||^2, ||F(x)||^2$',
                         legend=[[it.GetHRName() + ' $||x_n-y_n||^2$', it.GetHRName() + " $||F(x)||^2$"] for it in tested_items])


        plt.show()

# endregion

# grapher.plotFile('storage/methodstats/17_10_28_21_13_45/GradProj/stat.txt', xDataIndices=[0], yDataIndices=[2])

# AlgStatGrapher().plot(np.array(stat), xDataIndices=[0, 0, 0], yDataIndices=[[3], [3], [3]], plotTitle='Random $A$ 2000',
#                       xLabel='#iter.', yLabel='$||Ax-x||_L^2$',
#                       legend=[[type(it).__name__, type(it).__name__ + " D"] for it in tested_items])
