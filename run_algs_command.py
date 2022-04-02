import math
import datetime
import os
import io
import sys
import getopt
from typing import List, Union

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
from methods.IterativeAlgorithm import IterativeAlgorithm
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
from problems.problem import Problem
from problems.pseudomonotone_oper_one import PseudoMonotoneOperOne
from problems.pseudomonotone_oper_two import PseudoMonotoneOperTwo
from problems.sle_direct import SLEDirect
from problems.sle_saddle import SLESaddle

from problems.testcases import pseudo_mono_3, pseudo_mono_5, sle_saddle_hardcoded, sle_saddle_random_one, harker_test, \
    sle_saddle_regression_100_100000, minmax_game_1, pagerank_1

from problems.testcases.transport import pigu_sample, braess_sample, load_file_sample

from problems.testcases.slar_random import getSLE
from problems.viproblem import VIProblem
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
    adaptive_tau_small=0.45,
    show_plots=False
)

try:
    opts, args = getopt.getopt(sys.argv[1:], "ha:p:", ["algorithms=", "problem="])
except getopt.GetoptError:
    print('run_algs_command.py -a <algorithms> -p <problem>')
    exit(1)

problem_name = None
alg_names = []
for opt, arg in opts:
    if opt == '-h':
        print('run_algs_command.py -a <algorithms> -p <problem>')
        exit(0)
    elif opt in ("-a", "--algorithms"):
        alg_names = [a.lower() for a in arg.split(',')]
    elif opt in ("-p", "--problem"):
        problem_name = arg

print(f'Should run algorithms {alg_names} on problem {problem_name}')

if not problem_name or not alg_names:
    print(f'Both algorithms and problem should be specified!')
    exit(2)


def prepare_problem(problem_name: str) -> VIProblem:
    res: VIProblem = None
    if problem_name.lower() == 'pseudo_mono_3'.lower():
        res = pseudo_mono_3.prepareProblem(algorithm_params=params)
    elif problem_name.lower() == 'pseudo_mono_5'.lower():
        res = pseudo_mono_5.prepareProblem(algorithm_params=params)
    elif problem_name.lower() == 'harker_test'.lower():
        res = harker_test.prepareProblem(algorithm_params=params)
    elif problem_name.lower() == 'minmax_game_1'.lower():
        res = minmax_game_1.prepareProblem(algorithm_params=params)

    return res


def prepare_algs_to_test(alg_names: List[str], problem: VIProblem)->list[IterativeAlgorithm]:
    res: list[IterativeAlgorithm] = []

    if alg_names.count('Tseng'.lower()) > 0:
        res.append(
            Tseng(problem, stop_condition=params.stop_by,
                  eps=params.eps, lam=params.lam,
                  min_iters=params.min_iters, max_iters=params.max_iters, hr_name="Tseng")
        )

    if alg_names.count('MalitskyTam'.lower()) > 0:
        res.append(
            MalitskyTam(problem, stop_condition=params.stop_by,
                        x1=params.x1.copy(), eps=params.eps, lam=params.lam / 2.,
                        min_iters=params.min_iters, max_iters=params.max_iters, hr_name="MT")
        )

    if alg_names.count('ExtrapolationFromPast'.lower()) > 0:
        res.append(ExtrapolationFromPast(problem, stop_condition=params.stop_by,
                                         y0=params.x1.copy(), eps=params.eps, lam=params.lam*(math.sqrt(2.)-1),
                                         min_iters=params.min_iters, max_iters=params.max_iters, hr_name="EfP")
                   )

    if params.lam_spec_KL is not None or params.lam_spec is not None:
        lam_set_log: dict = {}
        for alg in res:
            algname: str = alg.__class__.__name__
            lam_dic: dict = params.lam_spec_KL if alg.projection_type == ProjectionType.BREGMAN else params.lam_spec
            if lam_dic is not None and algname in lam_dic:
                alg.lam = lam_dic[algname]
                lam_set_log[algname + ('-KL' if alg.projection_type == ProjectionType.BREGMAN else '')] = alg.lam

        if len(lam_set_log) > 0:
            print(f"Custom step sizes: {lam_set_log}")

    return res


captured_io = io.StringIO()
sys.stdout = captured_io

problem: VIProblem = prepare_problem(problem_name)

params.show_plots = False

# region Run all algs and save data and results
saved_history_dir = "storage/stats2022-04"
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

timings = {}
alg_history_list = []

if params.test_time:
    algs_to_test: List = None
    for i in range(params.test_time_count):
        algs_to_test = prepare_algs_to_test(alg_names, problem)
        for alg in algs_to_test:
            total_time: float = 0
            alg.do()
            if alg.isStopConditionMet() and alg.iter < params.max_iters:
                total_time = alg.history.iter_time_ns[alg.history.iters_count - 1]
            else:
                total_time = math.inf

            if alg.hr_name not in timings:
                timings[alg.hr_name] = {'time': 0.0}

            timings[alg.hr_name]['time'] += float(total_time)
            print(f"{i + 1}: {alg.hr_name} time: {total_time / params.time_scale_divider}s.")

            timings[alg.hr_name]['iter'] = alg.history.iters_count
            timings[alg.hr_name]['oper'] = alg.history.operator_count
            timings[alg.hr_name]['proj'] = alg.history.projections_count

    for alg in algs_to_test:
        timings[alg.hr_name]['time'] /= params.test_time_count
        timings[alg.hr_name]['time'] /= params.time_scale_divider

        BasicAlgoTests.PrintAlgRunStats(alg)

else:
    algs_to_test = prepare_algs_to_test(alg_names, problem)
    for alg in algs_to_test:
        alg.do()
        BasicAlgoTests.PrintAlgRunStats(alg)
        alg_history_list.append(alg.history)

        if params.save_history:
            df = alg.history.toPandasDF()
            df.to_excel(writer, sheet_name=alg.hr_name.replace("*", "_star"), index=False)
        print('')

        np.save('traff_eq_lastx', alg.history.x[alg.history.iters_count - 1])

if params.test_time:
    for k in timings:
        print(f"{k}: {timings[k]}")

if params.save_history:
    writer.save()
    writer.close()

sys.stdout = sys.__stdout__
# print(captured_io.getvalue())

f = open(os.path.join(saved_history_dir, f"log-{test_mnemo}.txt"), "w")
f.write(captured_io.getvalue())
f.close()

print(saved_history_dir)

# save history - takes too much space for big matrices!
# for idx, h in enumerate(alg_history_list):
#     hp = os.path.join(saved_history_dir, 'history', str(idx))
#     h.saveToDir(hp)

# endregion


# region Plot and save graphs
if params.save_plots or params.show_plots:
    grapher = AlgStatGrapher()
    grapher.plot_by_history(
        alg_history_list=alg_history_list,
        x_axis_type=params.x_axis_type, y_axis_type=params.y_axis_type, y_axis_label=params.y_label,
        styles=params.styles, start_iter=params.plot_start_iter,
        x_axis_label=params.x_label, time_scale_divider=params.time_scale_divider
    )

    if params.x_limits is not None:
        plt.xlim(params.x_limits)

    if params.y_limits is not None:
        plt.ylim(params.y_limits)

    if params.save_plots:
        dpi = 300.

        plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.svg"), bbox_inches='tight', dpi=dpi, format='svg')
        plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.eps"), bbox_inches='tight', dpi=dpi, format='eps')

        plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.png"), bbox_inches='tight', dpi=dpi)

    if params.show_plots:
        plt.title(problem.hr_name, loc='center')
        plt.show()

    exit(0)

# endregion


# table - time for getting to epsilon error
# for 1 and 2 - "real error"
# for 3 - step error (can be multiple solutions) and see F(x)
# show lambda on separate graph
