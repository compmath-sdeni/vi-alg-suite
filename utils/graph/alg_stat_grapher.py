from typing import List
from enum import Enum, unique

from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt

@unique
class XAxisType(Enum):
    ITERATION = 1
    TIME = 2

class AlgStatGrapher:
    defFont = 18

    def checkDataDims(self, dims: int, lst: list) -> bool:
        return dims == 2 if isinstance(lst[0], list) else dims == 1

    def initParamsArray(self, firstLevelDims: int, secondLevelDims: int, flatValsList: list) -> list:
        res = []
        k = 0

        for i in range(firstLevelDims):
            if firstLevelDims > 1:
                cur = []
                res.append(cur)
            else:
                cur = res

            for j in range(secondLevelDims):
                cur.append(flatValsList[k % len(flatValsList)])
                k += 1

        return res

    def plotFile(self, dataFile, *, xDataIndices: list = None, yDataIndices: list = None, graphColors: list = None,
                 legend: list = None, xLabel: str = 'Error', yLabel: str = 'Iterations', plotTitle: str = ''):
        data = np.loadtxt(dataFile)
        self.plot(data, xDataIndices=xDataIndices, yDataIndices=yDataIndices, graphColors=graphColors, legend=legend,
                  xLabel=xLabel, yLabel=yLabel, plotTitle=plotTitle)

    def plotSingleDim(self, *, data: np.ndarray, xDataIndex: int, yDataIndices: list, graphColors, legend, ax):
        for i in range(len(yDataIndices)):
            ax.plot(data[:, xDataIndex], data[:, yDataIndices[i]], graphColors[i % len(graphColors)],
                    label=legend[i] if legend is not None and i < len(legend) else str(i))

    def plot_by_history(self, *, alg_history_list: List[np.ndarray], x_axis_type: XAxisType = XAxisType.ITERATION,
                        plot_step_delta: bool = True, plot_real_error: bool = False,
                        x_axis_label: str = "Iteration", y_axis_label: str = "Error", plot_title: str = None,
                        legend: List[List[str]] = [], xScale: str = 'linear', yScale: str = 'log', start_iter: int = 2):
        y_dims: int = 0
        x_len: int = 0
        algs_count: int = len(alg_history_list)

        xDataIndices = []
        yDataIndices = []

        if x_axis_type == XAxisType.ITERATION:
            for alg_history in alg_history_list:
                xDataIndices.append([0])
                yDataIndices.append([1])

                if x_len < alg_history.iters_count - start_iter - 1:
                    x_len = alg_history.iters_count - start_iter - 1

        if plot_step_delta:
            y_dims += 1

        if plot_real_error:
            y_dims += 1

        # plot_data: np.ndarray = np.zeros((algs_count, x_len, y_dims+1), dtype=float)

        plot_legend: List[str] = legend[:]

        graph_colors = self.initParamsArray(algs_count, y_dims,
                                           ['g-', 'g--', 'b-', 'b--', 'r-', 'r--', 'c-', 'c--', 'm-', 'm--', 'k-', 'k--'])

        if plot_legend is None or len(plot_legend) == 0:
            plot_legend = self.initParamsArray(algs_count, y_dims, ['1', '2', '3', '4', '5'])

        rc('xtick', labelsize=self.defFont)
        rc('ytick', labelsize=self.defFont)

        fig, ax = plt.subplots(figsize=(16, 8), dpi=80)

        for i in range(algs_count):
            iters_count = alg_history_list[i].iters_count

            plot_data: np.ndarray = np.zeros((y_dims+1, iters_count - start_iter), dtype=float)
            plot_data[0] = np.arange(0, iters_count - start_iter)

            yDataIndices = []
            k = 1
            if plot_step_delta:
                plot_data[k] = alg_history_list[i].step_delta_norm[start_iter:iters_count]
                yDataIndices.append(k)
                k += 1

            if plot_real_error:
                plot_data[k] = alg_history_list[i].real_error[start_iter:iters_count]
                yDataIndices.append(k)
                k += 1

            self.plotSingleDim(
                data=plot_data.transpose(), xDataIndex=0, yDataIndices=yDataIndices, graphColors=graph_colors[i], legend=plot_legend[i], ax=ax
            )

        ax.set_xscale(xScale)
        ax.set_yscale(yScale)

        plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
                    ncol=2, shadow=False, title="", fancybox=False, fontsize=self.defFont)

        plt.xlabel(x_axis_label, fontsize=self.defFont + 2)
        plt.ylabel(y_axis_label, fontsize=self.defFont + 2)

        if plot_title is not None:
            plt.title(plot_title, loc='center')

        fig.tight_layout()

    def plot(self, data: np.ndarray, *, xDataIndices: list = None, yDataIndices: list = None, graphColors: list = None,
             legend: list = None,
             xLabel: str = 'Iterations', yLabel: str = 'Error', plotTitle: str = '', xScale: str = 'linear',
             yScale: str = 'log'):

        if xDataIndices is None or (isinstance(xDataIndices, list) and not xDataIndices):
            xDataIndices = [0]

        if not isinstance(xDataIndices, list):
            xDataIndices = [xDataIndices]

        dims = 2 if isinstance(xDataIndices, list) else 1

        if yDataIndices is None:
            if dims == 1:
                yDataIndices = [1]
            else:
                yDataIndices = []
                for i in range(dims):
                    yDataIndices.append([1])

        if not self.checkDataDims(dims, yDataIndices):
            raise ValueError('Wrong y data dimentions!')

        if graphColors is None:
            graphColors = self.initParamsArray(data.shape[0] if dims > 1 else 1,
                                               len(yDataIndices) if dims == 1 else len(yDataIndices[0]),
                                               ['g-', 'b--', 'r:', 'y-.', 'c-', 'm--', 'k:'])

        if not isinstance(graphColors[0], list):
            graphColors = [graphColors]

        if legend is None:
            legend = self.initParamsArray(data.shape[0] if dims > 1 else 1,
                                          len(yDataIndices) if dims == 1 else len(yDataIndices[0]),
                                          ['1', '2', '3', '4', '5'])

        if not isinstance(legend[0], list):
            legend = [legend]

        rc('xtick', labelsize=self.defFont)
        rc('ytick', labelsize=self.defFont)

        fig, ax = plt.subplots(figsize=(16, 8), dpi=80)

        for j in range(1 if dims == 1 else data.shape[0]):
            self.plotSingleDim(data=data if dims == 1 else np.array(data[j]), xDataIndex=xDataIndices[j],
                               yDataIndices=yDataIndices if dims == 1 else yDataIndices[j], graphColors=graphColors[j],
                               legend=legend[j], ax=ax)

        ax.set_xscale(xScale)
        ax.set_yscale(yScale)

        plt.legend(loc="upper right", bbox_to_anchor=[1, 1],
                   ncol=2, shadow=False, title="", fancybox=False, fontsize=self.defFont)

        plt.xlabel(xLabel, fontsize=self.defFont + 2)
        plt.ylabel(yLabel, fontsize=self.defFont + 2)

        if plotTitle is not None:
            plt.title(plotTitle, loc='center')

        fig.tight_layout()
