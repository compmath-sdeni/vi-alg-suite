import time
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List

from constraints.classic_simplex import ClassicSimplex
from methods.korpele_mod import KorpelevichMod
from methods.varistepthree import VaristepThree
from methods.varisteptwo import VaristepTwo
from problems.harker_test import HarkerTest
from problems.koshima_shindo import KoshimaShindo
from problems.matrix_oper_vi import MatrixOperVI
from problems.nonlin_r2_oper import NonlinR2Oper
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

#T = HarkerTest(4)
#exit(1)

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
        stat[statIdx].append([iterNum, timeElapsed, currentState['D'], currentState['F'][0]])
    return True


def allAlgsFinished():
    global stat
    global statIdx
    print('All tests finished.')


# endregion

# region default parameters
lam = 0.000001
eps = 1e-10

tet = 0.9
tau = 0.75
sigma = 1.0
stab = 0

minIterTime = 0
printIterEvery = 0
maxIters = 4000
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
#               C=Hyperrectangle(3, [(-10, 0.6), (0.3, 13), (-1.5, 13)]),
#               x0=np.array([2, 2, -7]),
#               xtest=np.array([1, 0, 0]),
#               L=10,
#               vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)],
#               hr_name='$(x-1)^2+y^2+(z+2)^2->min, C = [-10,0.6]x[0.3,13]x[-1.5,13]$'
#               )
# )

# dim = 100
# A = np.identity(dim, dtype=float)
# #A = np.random.rand(dim, dim)
# A[0,3] = 1
# A[3,0] = 1
# A[dim-1,7] = -1
# A[7,dim-1] = -1
# testX = np.ones(dim, dtype=float)
# x0 = np.random.rand(dim)
#
# problems.append(
#     MatrixOperVI(A=A, b=A @ testX, x0=x0,
#                  hr_name='$Ax=b$')
# )

# problems.append(
#     NonlinR2Oper(x0=np.array([1,1]), hr_name='$NonLinA$')
# )

N = 2000

#hr_bounds = [(-5,5) for i in range(N)]
#problems.append(HarkerTest(N, C=Hyperrectangle(N, hr_bounds), hr_name='HPHard'),)

ht = HarkerTest(N, C=ClassicSimplex(N, N), hr_name='HarkerTest', x0=np.ones(N))
problems.append(ht,)
lam = 0.4/ht.norm

print('Lam: ', lam)

# rotP = getProblem(N, False, False)
# rotP.hr_name = 'Приклад 1'
# problems.append(rotP)

# problems.append(KoshimaShindo(x0=np.array([5,1,1,1])))

# endregion

# region tests and plots
for p in problems:

    grad_desc = GradProj(p, eps, lam, min_iters=minIters)
    popov_subgrad = PopovSubgrad(p, eps, lam, min_iters=minIters)
    popov_subgrad.hr_name = 'Popov'
    korpele_basic = Korpelevich(p, eps, lam, min_iters=minIters)
    korpele_basic.hr_name='EGM'
    korpele_mod = KorpelevichMod(p, eps, lam, min_iters=minIters)
    varistepone = VaristepOne(p, eps, lam, min_iters=minIters)
    varistepone.hr_name = 'Alg1'

    # varisteptwo = VaristepTwo(p, eps, lam, min_iters=minIters, xstar=np.array([1.2247, 0, 0, 2.7753]))
    # varistepthree = VaristepThree(p, eps, lam, min_iters=minIters, xstar=np.array([1.2247, 0, 0, 2.7753]))

    varisteptwo = VaristepTwo(p, eps, lam, min_iters=minIters, xstar=np.zeros(N))
    varisteptwo.hr_name = 'Alg2'
    varistepthree = VaristepThree(p, eps, lam, min_iters=minIters)
    varistepthree.hr_name = 'Alg3'

    tryTimestamp = datetime.now()
    statIdx = 0
    stat = []

    tested_items = [
        varistepone
        ,varisteptwo
        #,varistepthree
        ,korpele_basic
        #,korpele_mod
        ,popov_subgrad
        #,grad_desc
    ]

    alghoTester = BasicAlghoTests(print_every=printIterEvery, max_iters=maxIters, min_time=minIterTime,
                                  on_alg_start=onAlgStart, on_alg_finish=onAlgFinish, on_iteration=onIter)

    res = alghoTester.DoTests(tested_items)

    grapher = AlgStatGrapher()

    if statIdx > 0:
        grapher.plot(np.array(stat), xDataIndices=[1 for i in range(len(tested_items))], yDataIndices=[[2] for i in range(len(tested_items))],
                     plotTitle=p.GetHRName(), xLabel='Час, c.', yLabel='$||x_{n}-x^*||^2$',
                     legend=[[it.GetHRName(), it.GetHRName() + " D"] for it in tested_items])

        grapher.plot(np.array(stat), xDataIndices=[0 for i in range(len(tested_items))], yDataIndices=[[2] for i in range(len(tested_items))],
                     plotTitle=p.GetHRName(), xLabel='Кількість ітерацій $n$', yLabel='$||x_{n}-x^*||^2$',
                     legend=[[it.GetHRName(), it.GetHRName() + " D"] for it in tested_items])

plt.show()

# endregion

# grapher.plotFile('storage/methodstats/17_10_28_21_13_45/GradProj/stat.txt', xDataIndices=[0], yDataIndices=[2])

# AlgStatGrapher().plot(np.array(stat), xDataIndices=[0, 0, 0], yDataIndices=[[3], [3], [3]], plotTitle='Random $A$ 2000',
#                       xLabel='#iter.', yLabel='$||Ax-x||_L^2$',
#                       legend=[[type(it).__name__, type(it).__name__ + " D"] for it in tested_items])
