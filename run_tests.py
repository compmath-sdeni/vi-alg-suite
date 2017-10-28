import time
import numpy as np
import os
from datetime import datetime

from methods.IterativeAlgorithm import IterativeAlgorithm
from problems.visual_params import VisualParams
from utils.graph.alg_stat_grapher import AlgStatGrapher
from methods.grad_proj import GradProj
from constraints.hyperrectangle import Hyperrectangle
from problems.funcndmin import FuncNDMin
from utils.test_alghos import BasicAlghoTests

print("Started.")

# region default parameters
lam = 0.01
eps = 1e-8

tet = 0.9999999
tau = 0.5
sigma = 1
stab = 0

minIterTime = 0
printIterEvery = 1
maxIters = 20000

# region 4d funcmin №1 ###############################################
p = FuncNDMin(3,
              lambda x: (x[0] - 1) ** 2 + x[1] ** 2 + x[2] ** 2,
              lambda x: np.array([2 * x[0] - 2, 2 * x[1], 2 * x[2]]),
              C=Hyperrectangle(3, [(-10, 3), (-13, 13), (-17, 13)]),
              x0=np.array([2, 2, -7]),
              xtest=np.array([1, 0, 0]),
              L=10,
              vis=[VisualParams(xl=-3, xr=3, yb=-3, yt=3, zn=0, zf=56, elev=22, azim=-49)]
              )
# endregion ###############################################

grad_desc = GradProj(p, eps, lam)

dataPath = 'storage/methodstats'
tryTimestamp = datetime.now()
statIdx = 0
stat = []

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
    if iterNum>0 :
        stat[statIdx].append([iterNum, timeElapsed, currentState['D'], currentState['F']])
    return True


alghoTester = BasicAlghoTests(print_every=printIterEvery, max_iters=maxIters, min_time=minIterTime,
                         on_alg_start=onAlgStart, on_alg_finish=onAlgFinish, on_iteration=onIter)


def allAlgsFinished():
    global stat
    global statIdx
    print('All tests finished.')

res = alghoTester.DoTests([grad_desc])

grapher = AlgStatGrapher()

grapher.plotFile('storage/methodstats/17_10_28_21_13_45/GradProj/stat.txt', xDataIndices=[0], yDataIndices=[2])

# AlgStatGrapher().plot(np.array(stat), xDataIndices=[0, 0, 0], yDataIndices=[[3], [3], [3]], plotTitle='Random $A$ 2000',
#                       xLabel='#iter.', yLabel='$||Ax-x||_L^2$',
#                       legend=[[type(it).__name__, type(it).__name__ + " D"] for it in tested_items])
