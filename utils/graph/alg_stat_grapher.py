from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt


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

    def plot(self, data: np.ndarray, *, xDataIndices: list = None, yDataIndices: list = None, graphColors: list = None,
             legend: list = None,
             xLabel: str = 'Iterations', yLabel: str = 'Error', plotTitle: str = '', xScale: str = 'linear', yScale: str = 'log'):

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
            graphColors = self.initParamsArray(data.shape[0] if dims>1 else 1, len(yDataIndices) if dims == 1 else len(yDataIndices[0]),
                                               ['g-', 'b--', 'r:', 'y-.', 'c-', 'm--', 'k:'])

        if not isinstance(graphColors[0], list):
            graphColors = [graphColors]

        if legend is None:
            legend = self.initParamsArray(data.shape[0] if dims>1 else 1, len(yDataIndices) if dims == 1 else len(yDataIndices[0]),
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