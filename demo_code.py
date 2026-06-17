params = AlgoParam(eps=1e-8, lam0=0.5, tau=0.75, hist=True)
# ..........
problem = prnk.prepare(algorithm_params=params, ...)
efp_alg = EfPAdapt(problem, stop=params.stop_by, ...)
# ..........
algs = [alg1a, alg2a, ...]
# ..........
for alg in algs_to_test:
    alg.do()
    AlgTest.PrintStats(alg, max_print_len=20)
    alg_history_list.append(alg.history)
    # Save history to excel

grapher = AlgStatGrapher()
grapher.plot_by_history(...)
# Save plots as png and eps