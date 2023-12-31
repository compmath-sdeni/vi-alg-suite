import math
import datetime
import os
import io
import sys
import time
from typing import List, Dict
import getopt

import numpy as np
import cvxpy as cp
import pandas
import pandas as pd
from matplotlib import pyplot as plt

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.classic_simplex import ClassicSimplex
from constraints.halfspace import HalfSpace
from constraints.hyperplane import Hyperplane
from constraints.l1_ball import L1Ball
from methods.IterGradTypeMethod import ProjectionType, IterGradTypeMethod
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

from problems.nagurna_simplest import BloodDeliveryHardcodedOne

from problems.testcases import pseudo_mono_3, pseudo_mono_5, sle_saddle_hardcoded, sle_saddle_random_one, harker_test, \
    sle_saddle_regression_100_100000, pagerank_1, func_nd_min_mean_linear

from problems.funcndmin import FuncNDMin

from problems.testcases.zero_sum_game import minmax_game_1, minmax_game_2, blotto_game, minmax_game_test_1

from problems.testcases.transport import \
    pigu_sample, braess_sample, load_file_sample, test_one_sample, test_two_sample, test_three_sample, test_sample_1_2, \
    test_sample_1_3

from problems.testcases.slar_random import getSLE

from problems.testcases.blood_delivery import blood_delivery_hardcoded_test_one, blood_delivery_test_one, \
    blood_delivery_test_two, blood_delivery_test_three
from problems.viproblem import VIProblem

from utils.alg_history import AlgHistory
from utils.graph.alg_stat_grapher import AlgStatGrapher, XAxisType, YAxisType

from constraints.hyperrectangle import Hyperrectangle

from methods.korpelevich import Korpelevich
from utils.test_alghos import BasicAlgoTests


class AlgsRunner:
    def __init__(self, *, problem: VIProblem = None, params: AlgorithmParams = None, show_output: bool = True,
                 runs_data_save_path: str = 'storage/alg_run_stats'):

        self.available_algs: List[Dict[(str, IterGradTypeMethod)]] = None
        self.available_algs_dict: Dict[str, IterGradTypeMethod] = None
        self.problem: VIProblem = problem

        self.runs_data_save_path: str = runs_data_save_path
        os.makedirs(self.runs_data_save_path, exist_ok=True)

        if params is None:
            self.params: AlgorithmParams = AlgorithmParams(
                eps=1e-5,
                min_iters=10,
                max_iters=500,
                lam=0.01,
                lam_KL=0.005,
                start_adaptive_lam=0.5,
                start_adaptive_lam1=0.5,
                adaptive_tau=0.75,
                adaptive_tau_small=0.45,
                save_history=True,
                excel_history=True
            )
        else:
            self.params: AlgorithmParams = params

        self.show_output: bool = show_output

    def set_problem(self, problem: VIProblem):
        self.problem = problem

    def prepare_predefined_problem(self, problem_name: str):
        res: VIProblem = None

        if problem_name.lower() == 'pseudo_mono_3'.lower():
            res = pseudo_mono_3.prepareProblem(algorithm_params=self.params)
        elif problem_name.lower() == 'pseudo_mono_5'.lower():
            res = pseudo_mono_5.prepareProblem(algorithm_params=self.params)
        elif problem_name.lower() == 'harker_test'.lower():
            res = harker_test.prepareProblem(algorithm_params=self.params)
        elif problem_name.lower() == 'minmax_game_1'.lower():
            res = minmax_game_1.prepareProblem(algorithm_params=self.params)
        elif problem_name.lower() == 'blood_supply'.lower():
            res = blood_delivery_test_three.prepareProblem(algorithm_params=self.params)

        self.problem = res

        return self.problem

    @staticmethod
    def get_avail_alg_names():
        return ['Korpelevich', 'Korpelevich (A)', 'Tseng', 'Tseng (A)', 'Tseng (A) - KL', 'Tseng - KL', 'EfP',
                'EfP (A)', 'EfP (A) - KL', 'EFP - KL', 'MT', 'MT (A)', 'MT (A) - KL', 'MT - KL']

    def init_algs(self):
        korpele = Korpelevich(self.problem, eps=self.params.eps, lam=self.params.lam, min_iters=self.params.min_iters,
                              max_iters=self.params.max_iters, hr_name="Kor")
        korpele_adapt = KorpelevichMod(self.problem, eps=self.params.eps, min_iters=self.params.min_iters,
                                       max_iters=self.params.max_iters,
                                       hr_name="Kor (A)")

        tseng = Tseng(self.problem, stop_condition=self.params.stop_by,
                      eps=self.params.eps, lam=self.params.lam,
                      min_iters=self.params.min_iters, max_iters=self.params.max_iters, hr_name="Tseng")

        tseng_bregproj = Tseng(self.problem, stop_condition=self.params.stop_by,
                               eps=self.params.eps,
                               lam=self.params.lam_KL,
                               min_iters=self.params.min_iters, max_iters=self.params.max_iters,
                               hr_name="Tseng - KL", projection_type=ProjectionType.BREGMAN)

        tseng_adaptive = TsengAdaptive(self.problem,
                                       eps=self.params.eps, lam=self.params.start_adaptive_lam,
                                       tau=self.params.adaptive_tau,
                                       save_history=self.params.save_history,
                                       min_iters=self.params.min_iters, max_iters=self.params.max_iters,
                                       hr_name="Tseng (A)")

        tseng_adaptive_bregproj = TsengAdaptive(self.problem, stop_condition=self.params.stop_by,
                                                eps=self.params.eps, lam=self.params.start_adaptive_lam,
                                                tau=self.params.adaptive_tau,
                                                min_iters=self.params.min_iters, max_iters=self.params.max_iters,
                                                hr_name="Alg. 1* (A)", projection_type=ProjectionType.BREGMAN)

        extrapol_from_past = ExtrapolationFromPast(self.problem, stop_condition=self.params.stop_by,
                                                   y0=self.params.x1.copy(), eps=self.params.eps,
                                                   lam=self.params.lam * (math.sqrt(2.) - 1),
                                                   min_iters=self.params.min_iters, max_iters=self.params.max_iters,
                                                   hr_name="EfP")

        extrapol_from_past_bregproj = ExtrapolationFromPast(self.problem, stop_condition=self.params.stop_by,
                                                            y0=self.params.x1.copy(), eps=self.params.eps,
                                                            lam=self.params.lam_KL * (math.sqrt(2.) - 1),
                                                            min_iters=self.params.min_iters,
                                                            max_iters=self.params.max_iters,
                                                            hr_name="EFP - KL", projection_type=ProjectionType.BREGMAN)

        extrapol_from_past_adaptive = ExtrapolationFromPastAdapt(self.problem, stop_condition=self.params.stop_by,
                                                                 y0=self.params.x1.copy(), eps=self.params.eps,
                                                                 lam=self.params.start_adaptive_lam,
                                                                 tau=self.params.adaptive_tau_small,
                                                                 min_iters=self.params.min_iters,
                                                                 max_iters=self.params.max_iters,
                                                                 hr_name="EFP (A)")

        extrapol_from_past_adaptive_bregproj = ExtrapolationFromPastAdapt(self.problem,
                                                                          stop_condition=self.params.stop_by,
                                                                          y0=self.params.x1.copy(), eps=self.params.eps,
                                                                          lam=self.params.start_adaptive_lam1,
                                                                          tau=self.params.adaptive_tau_small,
                                                                          min_iters=self.params.min_iters,
                                                                          max_iters=self.params.max_iters,
                                                                          hr_name="EFP (A) - KL",
                                                                          projection_type=ProjectionType.BREGMAN)

        malitsky_tam = MalitskyTam(self.problem, stop_condition=self.params.stop_by,
                                   x1=self.params.x1.copy(), eps=self.params.eps, lam=self.params.lam / 2.,
                                   min_iters=self.params.min_iters, max_iters=self.params.max_iters, hr_name="MT")

        malitsky_tam_bregproj = MalitskyTam(self.problem, stop_condition=self.params.stop_by,
                                            x1=self.params.x1.copy(), eps=self.params.eps, lam=self.params.lam_KL / 2.,
                                            min_iters=self.params.min_iters, max_iters=self.params.max_iters,
                                            hr_name="MT - KL", projection_type=ProjectionType.BREGMAN)

        malitsky_tam_adaptive = MalitskyTamAdaptive(self.problem,
                                                    x1=self.params.x1.copy(), eps=self.params.eps,
                                                    stop_condition=self.params.stop_by,
                                                    lam=self.params.start_adaptive_lam,
                                                    lam1=self.params.start_adaptive_lam,
                                                    tau=self.params.adaptive_tau,
                                                    min_iters=self.params.min_iters, max_iters=self.params.max_iters,
                                                    hr_name="MT (A)")

        malitsky_tam_adaptive_bregproj = MalitskyTamAdaptive(self.problem,
                                                             x1=self.params.x1.copy(), eps=self.params.eps,
                                                             stop_condition=self.params.stop_by,
                                                             lam=self.params.start_adaptive_lam1,
                                                             lam1=self.params.start_adaptive_lam1,
                                                             tau=self.params.adaptive_tau,
                                                             min_iters=self.params.min_iters,
                                                             max_iters=self.params.max_iters,
                                                             hr_name="MT (A) - KL",
                                                             projection_type=ProjectionType.BREGMAN)

        self.available_algs = [
            {"name": "Korpelevich", "alg": korpele},
            {"name": "Korpelevich (A)", "alg": korpele_adapt},
            {"name": "Tseng", "alg": tseng},
            {"name": "Tseng (A)", "alg": tseng_adaptive},
            {"name": "Tseng (A) - KL", "alg": tseng_adaptive_bregproj},
            {"name": "Tseng - KL", "alg": tseng_bregproj},
            {"name": "EfP", "alg": extrapol_from_past},
            {"name": "EfP (A)", "alg": extrapol_from_past_adaptive},
            {"name": "EfP (A) - KL", "alg": extrapol_from_past_adaptive_bregproj},
            {"name": "EFP - KL", "alg": extrapol_from_past_bregproj},
            {"name": "MT", "alg": malitsky_tam},
            {"name": "MT (A)", "alg": malitsky_tam_adaptive},
            {"name": "MT (A) - KL", "alg": malitsky_tam_adaptive_bregproj},
            {"name": "MT - KL", "alg": malitsky_tam_bregproj}
        ]

        # create a dict for fast access
        self.available_algs_dict = {alg["name"]: alg["alg"] for alg in self.available_algs}

        if self.params.lam_spec_KL is not None or self.params.lam_spec is not None:
            lam_set_log: dict = {}
            for name, alg in self.available_algs:
                algname: str = alg.__class__.__name__
                lam_dic: dict = self.params.lam_spec_KL if alg.projection_type == ProjectionType.BREGMAN else self.params.lam_spec
                if lam_dic is not None and algname in lam_dic:
                    alg.lam = lam_dic[algname]
                    lam_set_log[algname + ('-KL' if alg.projection_type == ProjectionType.BREGMAN else '')] = alg.lam

            if len(lam_set_log) > 0:
                print(f"Custom step sizes: {lam_set_log}")

        return self.available_algs

    def run_algs(self, algs_to_test_names: List[str], *, show_output: bool = True):

        result = {}

        self.captured_io = io.StringIO()
        sys.stdout = self.captured_io

        start = time.monotonic()

        saved_history_dir = f"{self.runs_data_save_path}/{datetime.datetime.today().strftime('%Y-%m')}"
        test_mnemo = f"{self.problem.GetUniqueName()}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        saved_history_dir = os.path.join(saved_history_dir, test_mnemo)
        os.makedirs(saved_history_dir, exist_ok=True)

        result["run_results_path"] = saved_history_dir

        self.problem.saveToDir(path_to_save=os.path.join(saved_history_dir, "problem"))
        self.params.saveToDir(os.path.join(saved_history_dir, "params"))

        result["problem_folder"] = "problem"
        result["params_folder"] = "params"

        print(f"Problem: {self.problem.GetFullDesc()}")
        print(f"Algorithms: {algs_to_test_names}")

        print()

        print(f"eps: {self.params.eps}; tau1: {self.params.adaptive_tau}; tau2: {self.params.adaptive_tau_small}; "
              f"start_lam: {self.params.start_adaptive_lam}; start_lam1: {self.params.start_adaptive_lam1}; "
              f"lam: {self.params.lam}; lam_KL: {self.params.lam_KL}")

        print()

        if self.params.save_history and self.params.excel_history:
            writer = pd.ExcelWriter(
                os.path.join(saved_history_dir, f"history-{test_mnemo}.xlsx"),
                engine='xlsxwriter')

        timings = {}
        alg_history_list = []

        if self.params.test_time:
            algs_to_test: List = None
            for i in range(self.params.test_time_count):
                algs_to_test = [self.available_algs_dict[alg_name] for alg_name in algs_to_test_names]
                for alg in algs_to_test:
                    total_time: float = 0
                    alg.do()
                    if alg.isStopConditionMet() and alg.iter < self.params.max_iters:
                        total_time = alg.history.iter_time_ns[alg.history.iters_count - 1]
                    else:
                        total_time = math.inf

                    if alg.hr_name not in timings:
                        timings[alg.hr_name] = {'time': 0.0}

                    timings[alg.hr_name]['time'] += float(total_time)
                    print(f"{i + 1}: {alg.hr_name} time: {total_time / self.params.time_scale_divider}s.")

                    timings[alg.hr_name]['iter'] = alg.history.iters_count
                    timings[alg.hr_name]['oper'] = alg.history.operator_count
                    timings[alg.hr_name]['proj'] = alg.history.projections_count

            for alg in algs_to_test:
                timings[alg.hr_name]['time'] /= self.params.test_time_count
                timings[alg.hr_name]['time'] /= self.params.time_scale_divider

                BasicAlgoTests.PrintAlgRunStats(alg)

        else:
            algs_to_test = [self.available_algs_dict[alg_name] for alg_name in algs_to_test_names]
            for alg in algs_to_test:
                try:
                    alg.do()
                except Exception as e:
                    print(e)

                BasicAlgoTests.PrintAlgRunStats(alg)
                alg_history_list.append(alg.history)

                if self.params.save_history:

                    # save to excel
                    if self.params.excel_history:
                        # formatting params - precision is ignored for some reason...
                        with np.printoptions(threshold=500, precision=3, edgeitems=10, linewidth=sys.maxsize,
                                             floatmode='fixed'):
                            df: pandas.DataFrame = alg.history.toPandasDF()
                            df.to_excel(writer, sheet_name=alg.hr_name.replace("*", "_star"), index=False)

                    # save to csv without cutting data (without ...)
                    with np.printoptions(threshold=sys.maxsize, precision=3, linewidth=sys.maxsize, floatmode='fixed'):
                        # pandas.set_option('display.max_columns', None)
                        # pandas.set_option('display.max_colwidth', None)
                        # pandas.set_option('display.max_seq_items', None)

                        df: pandas.DataFrame = alg.history.toPandasDF()

                        history_csv_path = os.path.join(saved_history_dir, f"history-{test_mnemo}.csv")
                        df.to_csv(history_csv_path)

                        result["history_csv_path"] = "history_csv_path"

                print('')

                # save last approximation - to start from it next time
                # np.save('traff_eq_lastx', alg.history.x[alg.history.iters_count - 1])

        if self.params.test_time:
            print(
                f"Averaged time over {self.params.test_time_count} runs. Stop conditions: {self.params.stop_by} with epsilon {self.params.eps}")
            for k in timings:
                print(f"{k}: {timings[k]}")

        if self.params.save_history and self.params.excel_history:
            # save excel file
            # writer.save()
            writer.close()

        sys.stdout = sys.__stdout__

        if self.show_output:
            print(self.captured_io.getvalue())

        run_log_file_path = os.path.join(saved_history_dir, f"log-{test_mnemo}.txt")

        f = open(run_log_file_path, "w")
        f.write(self.captured_io.getvalue())
        f.close()

        result["run_log_file_path"] = run_log_file_path

        # save history - takes too much space for big matrices!
        # for idx, h in enumerate(alg_history_list):
        #     hp = os.path.join(saved_history_dir, 'history', str(idx))
        #     h.saveToDir(hp)

        finish = time.monotonic()

        final_timing_str = f'Total time of calculation, history saving etc. (without plotting): {(finish - start):.0f} sec.'

        print(final_timing_str)

        f = open(os.path.join(saved_history_dir, f"log-{test_mnemo}.txt"), "a+")
        f.write(final_timing_str)
        f.close()

        # endregion

        # region Plot and save graphs
        if self.params.save_plots or self.params.show_plots:
            if not self.params.save_history:
                print("Graphics generation is impossible if history is not collected!")
            else:
                grapher = AlgStatGrapher()
                grapher.plot_by_history(
                    alg_history_list=alg_history_list,
                    x_axis_type=self.params.x_axis_type, y_axis_type=self.params.y_axis_type,
                    y_axis_label=self.params.y_label,
                    styles=self.params.styles, start_iter=self.params.plot_start_iter,
                    x_axis_label=self.params.x_label, time_scale_divider=self.params.time_scale_divider
                )

                if self.params.x_limits is not None:
                    plt.xlim(self.params.x_limits)

                if self.params.y_limits is not None:
                    plt.ylim(self.params.y_limits)

                if self.params.save_plots:
                    dpi = 300.

                    plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.svg"), bbox_inches='tight',
                                dpi=dpi,
                                format='svg')
                    plt.savefig(os.path.join(saved_history_dir, f"graph-{test_mnemo}.eps"), bbox_inches='tight',
                                dpi=dpi,
                                format='eps')

                    graph_results_path = os.path.join(saved_history_dir, f"graph-{test_mnemo}.png")
                    plt.savefig(graph_results_path, bbox_inches='tight', dpi=dpi)

                    result["graph_results"] = [graph_results_path]

                if self.params.show_plots:
                    plt.title(self.problem.hr_name, loc='center')
                    plt.show()

        # endregion

        return result


# table - time for getting to epsilon error
# for 1 and 2 - "real error"
# for 3 - step error (can be multiple solutions) and see F(x)
# show lambda on separate graph


if __name__ == "__main__":
    params_dict = {
        'eps': 1e-5,
        'min_iters': 10,
        'max_iters': 500,
        'lam': 0.01,
        'lam_KL': 0.005,
        'start_adaptive_lam': 0.5,
        'start_adaptive_lam1': 0.5,
        'adaptive_tau': 0.75,
        'adaptive_tau_small': 0.45,
        'save_history': True,
        'excel_history': True
    }

    params = AlgorithmParams(**params_dict)
    problem = blood_delivery_test_three.prepareProblem(algorithm_params=params, show_network=False, print_data=False)
    alg_names = ['Tseng', 'MT']

    runner = AlgsRunner(problem=problem, params=params, runs_data_save_path='storage/alg_run_stats/local_test_runs')

    # runner.prepare_predefined_problem('minmax_game_1')
    # runner.prepare_predefined_problem('blood_supply')

    runner.init_algs()
    runner.run_algs(alg_names)
