from typing import List
from enum import Enum, unique

import numpy as np
from matplotlib import cm
from matplotlib import rc
import matplotlib.pyplot as plt


class AlgStatGrapher2:
    def plot_by_history(self, *, alg_history_list: List[np.ndarray], plot_step_delta: bool = True, plot_real_error: bool = False,
                        x_axis_label: str = "Iteration", y_axis_label: str = "Error",
                        legend: List[List[str]] = [], xScale: str = 'linear', yScale: str = 'log', start_iter: int = 2):
        graph_colors = ['g-', 'b--', 'r:', 'y-.', 'c-', 'm--', 'k:']
        plot_legend: List[str] = legend[:]

        rc('xtick', labelsize=18)
        rc('ytick', labelsize=18)

        fig, ax = plt.subplots(figsize=(16, 8), dpi=80)

        N = 100

        plot_data: np.ndarray = np.zeros((4, N), dtype=float)
        plot_data[0] = np.arange(0, N)
        plot_data[1] = np.arange(0, N)
        plot_data[2] = np.arange(2, N + 2)
        plot_data[3] = np.arange(5, N + 5)

        ax.plot(plot_data[0], plot_data[1], graph_colors[0])
        ax.plot(plot_data[0], plot_data[2], graph_colors[2])
        ax.plot(plot_data[0], plot_data[3], graph_colors[4])
        plt.legend(labels=plot_legend)

        plt.show()

        exit()
