import numpy as np
from matplotlib import pyplot as plt

from methods.korpele_mod import KorpelevichMod
from methods.malitsky_tam import MalitskyTam
from methods.malitsky_tam_adaptive import MalitskyTamAdaptive
from methods.tseng import Tseng
from methods.tseng_adaptive import TsengAdaptive
from utils.graph.alg_stat_grapher import AlgStatGrapher

from constraints.hyperrectangle import Hyperrectangle

from methods.korpelevich import Korpelevich
from problems.funcndmin import FuncNDMin
from utils.test_alghos import BasicAlgoTests

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

# region (X1+X2+...+Xn - n/2)^2 -> min; lam = 1/4N

N = 100

x0 = np.array([i+1 for i in range(N)])
x1 = np.array([1 for i in range(N)])
def_lam = 0.0001

real_solution = np.array([0.5 for i in range(N)])

problem = FuncNDMin(N,
              lambda x: (np.sum(x) - N/2) ** 2,
              lambda x: np.ones(N) * 2 * (np.sum(x) - N/2),
              C=Hyperrectangle(N, [(0, 5) for i in range(N)]),
              x0=x0,
              hr_name='$(x_1 + x_2 + ... + x_n - n/2)^2->min, C = [-5,5]x[-5,5], N = {0}$'.format(N)
              )

# endregion


min_iters = 10
max_iters = 2000

korpele = Korpelevich(problem, eps=0.0000001, lam=def_lam, min_iters=min_iters, max_iters=max_iters)
korpele_adapt = KorpelevichMod(problem, eps=0.00001, min_iters=min_iters, max_iters=max_iters)

tseng = Tseng(problem, eps=0.0000001, lam=def_lam, min_iters=min_iters, max_iters=max_iters)
tseng_adaptive = TsengAdaptive(problem, eps=0.0000001, lam=def_lam, min_iters=min_iters, max_iters=max_iters)

malitsky_tam = MalitskyTam(problem, x1=x1.copy(), eps=0.0000001, lam=def_lam, min_iters=min_iters, max_iters=max_iters)
malitsky_tam_adaptive = MalitskyTamAdaptive(problem, x1=x1.copy(), eps=0.0000001, lam=def_lam, min_iters=min_iters, max_iters=max_iters)

korpele.do()
BasicAlgoTests.PrintAlgRunStats(korpele)

korpele_adapt.do()
BasicAlgoTests.PrintAlgRunStats(korpele_adapt)

tseng.do()
BasicAlgoTests.PrintAlgRunStats(tseng)

tseng_adaptive.do()
BasicAlgoTests.PrintAlgRunStats(tseng_adaptive)

malitsky_tam.do()
BasicAlgoTests.PrintAlgRunStats(malitsky_tam)

malitsky_tam_adaptive.do()
BasicAlgoTests.PrintAlgRunStats(malitsky_tam_adaptive)


alg_history_list = []
legends = []

alg_history_list.append(korpele.history)
legends.append(['Korpelevich', 'Korpelevich - real err'])

alg_history_list.append(korpele_adapt.history)
legends.append(['Adaptive Korpelevich', 'Ad. Korp. - real err'])

alg_history_list.append(tseng.history)
legends.append(['Tseng', 'Tseng - real err'])

alg_history_list.append(tseng_adaptive.history)
legends.append(['Adaptive Tseng', 'Ad Tseng - real err'])

alg_history_list.append(malitsky_tam.history)
legends.append(['MT', 'MT - real err'])

alg_history_list.append(malitsky_tam_adaptive.history)
legends.append(['MT adapt', 'MT adapt - real err'])

grapher = AlgStatGrapher()
grapher.plot_by_history(alg_history_list=alg_history_list, plot_step_delta=True, legend=legends, plot_real_error=True)

plt.show()

exit()
