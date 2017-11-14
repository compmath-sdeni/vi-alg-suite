import time
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List

from constraints.classic_simplex import ClassicSimplex
from methods.batched_grad_proj import BatchedGradProj
from methods.korpele_mod import KorpelevichMod
from methods.varistepthree import VaristepThree
from methods.varisteptwo import VaristepTwo
from problems.func_sum_min_simple import FuncSumMinSimple
from problems.harker_test import HarkerTest
from problems.koshima_shindo import KoshimaShindo
from problems.lin_sys_splitted import LinSysSplitted
from problems.lin_sys_splitted_l1 import LinSysSplittedL1
from problems.log_reg_flavor_one import LogRegFlavorOne
from problems.log_reg_flavor_two import LogRegFlavorTwo

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
    if iterNum >= 0:
        stat[statIdx].append([iterNum, timeElapsed, currentState['D'], currentState['F'][0], currentState['F'][1] if len(currentState['F'])>1 else currentState['F'][0]])
    return True


def allAlgsFinished():
    global stat
    global statIdx
    print('All tests finished.')


# endregion

# region default parameters
do_graph = True

lam = 0.001
eps = 1e-10
tet = 0.9
tau = 0.5
sigma = 1.0
stab = 0

minIterTime = 0
printIterEvery = 1
maxIters = 100000
minIters = 0

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

N = 3
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

# N = 3
#
# A = np.eye(N, N)
# b = np.ones(N)
#
# problems.append(
#     LinSysSplitted(N, A, b,
#               C=Hyperrectangle(N, [(-10, 10), (-10.3, 13), (-10.5, 13)]),
#               x0=np.array([2, 2, -7]),
#               xtest=np.array([1, 0, -2]),
#               vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#               hr_name='$Ax-b$'
#               )
# )

N = 40
np.random.seed(N)

if N == 3:
    A = np.array([
        [5, 2, 1]
        , [1, 13, 4]
        , [3, 7, 1]
    ], dtype=float)
else:
    A = np.random.rand(N, N)

A = A @ A.T

for i in range(N):
    A[i, i] += np.sum(A[i]) * 2.0

print(vectorToString(A))

norm = np.linalg.norm(A, 2)
lam = (1.0 / norm)
print("Norm: {0}; Lam: {1}".format(norm, lam))

# lam = 0.1

lam_batched = lam

maxIters = 500
printIterEvery = 1

xtest = np.array([i + 1 for i in range(N)])

# b = A @ np.ones(N, dtype=float)
b = A @ xtest
# x0 = np.ones(N)
x0 = (np.random.rand(N)*0.01 - 0.5*0.01) + xtest

print("Initial error 2: ", (np.dot(A@x0 - b, A@x0 - b)))
print("Initial error 1: ", (np.abs(A@x0 - b).sum()))

# problems.append(
#     LinSysSplitted(N, A, b,
#                    #C=Hyperrectangle(N, [(i+0.5, i+1.5) for i in range(N)]),
#                    C=Hyperrectangle(N, [(-100, 100) for i in range(N)]),
#                    x0=x0,
#                    xtest=xtest,
#                    vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#                    hr_name='$||Ax-b||_2$',
#                    lam_override=lam*lam
#                    )
# )

# lam = 0.00001
# problems.append(
#     LinSysSplittedL1(N, A, b,
#               C=Hyperrectangle(N, [(-100, 100) for i in range(N)]),
#               x0=x0,
#               xtest=xtest,
#               vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#               hr_name='$||Ax-b||_1$',
#               lam_override=lam*lam
#               )
# )

X = np.array([
    [1, 1, 1],
    [-1, 1, 1],
    [1, 1, -1],
    [-31, 1, -1],
    [31, 1, 1],
    [-12, 1, 1],
    [21, 1, -1],
    [-1, 45, 1],
    [-1, 4, 1],
    [-1, -31, 1],
    [1, 33, -1]
])

y = np.array([np.sign(t[-1]) for t in X])

print("LogRegCont: X: {0}; y:{1} ", (np.dot(A@x0 - b, A@x0 - b)))
print("Initial error 1: ", (np.abs(A@x0 - b).sum()))

wtest = np.array([0, 0, 1])
w0 = np.array([1., 1, 1.0])

problems.append(
    LogRegFlavorTwo(X, y,
                    C=Hyperrectangle(X.shape[1], [(-100, 100) for i in range(X.shape[1])]),
                    w0=w0,
                    wtest=wtest,
                    hr_name='LogResCont',
                    lam_override=0.01
                    )
)

# problems.append(
#     MatrixOperVI(A, b,
#               x0=np.array([1.3, 0.7, 1], dtype=float),
#               hr_name='$Ax-b$'
#               )
# )

# N = 100
# problems.append(
#     FuncNDMin(N,
#               lambda x: (np.sum(x) - N/2) ** 2,
#               lambda x: np.ones(N) * 2 * (np.sum(x) - N/2),
#               C=Hyperrectangle(N, [(-5, 5) for i in range(N)]),
#               x0=np.array([i+1 for i in range(N)]),
#               hr_name='$(x + y -1)^2->min, C = [-5,5]x[-5,5]$'
#               )
# )

# N = 100
# A = np.identity(N, dtype=float)
# #A = np.random.rand(dim, dim)
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

# ppr = PageRankProblem.CreateRandom(N, 0.01)
# problems.append(ppr)

# endregion

# region tests and plots
for p in problems:

    grad_desc = GradProj(p, eps, lam, min_iters=minIters)
    popov_subgrad = PopovSubgrad(p, eps, lam, min_iters=minIters)
    popov_subgrad.hr_name = 'Popov'
    korpele_basic = Korpelevich(p, eps, lam, min_iters=minIters)
    korpele_basic.hr_name = 'EGM'
    korpele_mod = KorpelevichMod(p, eps, lam, min_iters=minIters)
    varistepone = VaristepOne(p, eps, lam, min_iters=minIters)
    varistepone.hr_name = 'Alg1'

    # varisteptwo = VaristepTwo(p, eps, lam, min_iters=minIters, xstar=np.array([1.2247, 0, 0, 2.7753]))
    # varistepthree = VaristepThree(p, eps, lam, min_iters=minIters, xstar=np.array([1.2247, 0, 0, 2.7753]))
    xst = np.ones(N)
    # xst[0]=0

    varisteptwo = VaristepTwo(p, eps, lam, min_iters=minIters, xstar=xst, alfa_calc=lambda i: 1.0 / (i + 1))
    varisteptwo.hr_name = 'Alg2'
    varistepthree = VaristepThree(p, eps, lam, min_iters=minIters)
    varistepthree.hr_name = 'Alg3'

    batched_grad_proj = BatchedGradProj(p, eps, lam_batched, min_iters=minIters, split_count=1, hr_name='BatchedNoSplit')
    batched_grad_proj2 = BatchedGradProj(p, eps, lam_batched, min_iters=minIters, split_count=2, hr_name='Batched x2')
    batched_grad_proj4 = BatchedGradProj(p, eps, lam_batched, min_iters=minIters, split_count=4, hr_name='Batched x4')
    batched_grad_proj8 = BatchedGradProj(p, eps, lam_batched, min_iters=minIters, split_count=8, hr_name='Batched x8')

    tryTimestamp = datetime.now()
    statIdx = 0
    stat = []

    tested_items = [
        grad_desc
        ,
        batched_grad_proj
        #, batched_grad_proj2
        , batched_grad_proj4
        #, batched_grad_proj8
        # ,varistepone
        # ,varisteptwo
        # ,varistepthree
        # ,korpele_basic
        # ,korpele_mod
        # ,popov_subgrad
    ]

    alghoTester = BasicAlghoTests(print_every=printIterEvery, max_iters=maxIters, min_time=minIterTime,
                                  on_alg_start=onAlgStart, on_alg_finish=onAlgFinish, on_iteration=onIter)

    res = alghoTester.DoTests(tested_items)

    if do_graph:
        grapher = AlgStatGrapher()

        if statIdx > 0:
            grapher.plot(np.array(stat), xDataIndices=[1 for i in range(len(tested_items))],
                         yDataIndices=[[3, 4] for i in range(len(tested_items))],
                         plotTitle=p.GetHRName(), xLabel='Час, c.', yLabel='$||x_{n}-y_n||^2$',
                         legend=[[it.GetHRName() + ' $u_n=x_n$', it.GetHRName() + " $u_n=z_n$"] for it in tested_items])

            grapher.plot(np.array(stat), xDataIndices=[0 for i in range(len(tested_items))],
                         yDataIndices=[[3, 4] for i in range(len(tested_items))],
                         plotTitle=p.GetHRName(), xLabel='Кількість ітерацій $n$', yLabel='$||F(u_n)||^2$',
                         legend=[[it.GetHRName() + ' $u_n=x_n$', it.GetHRName() + " $u_n=z_n$"] for it in tested_items])


        plt.show()

# endregion

# grapher.plotFile('storage/methodstats/17_10_28_21_13_45/GradProj/stat.txt', xDataIndices=[0], yDataIndices=[2])

# AlgStatGrapher().plot(np.array(stat), xDataIndices=[0, 0, 0], yDataIndices=[[3], [3], [3]], plotTitle='Random $A$ 2000',
#                       xLabel='#iter.', yLabel='$||Ax-x||_L^2$',
#                       legend=[[type(it).__name__, type(it).__name__ + " D"] for it in tested_items])
