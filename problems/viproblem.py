from typing import Union
import numpy as np
from problems.problem import Problem


class VIProblem(Problem):
    def Project(self, x: Union[np.ndarray, float]):
        pass

    def Fcut1D(self, defx: Union[np.ndarray, float], x: Union[np.ndarray, float], dim: int):
        defx[dim] = x
        return self.F(defx)

    def Fcut2D(self, defx, x, y, xdim, ydim):
        defx[xdim] = x
        defx[ydim] = y
        return self.F(defx)

    def Name(self) -> str:
        pass

    def Draw2DProj(self, fig, ax, vis, xdim, ydim, curX=None, mutableOnly=False):
        return None

    def Draw3DProj(self, fig, ax, vis, xdim, ydim, curX=None):
        return None
