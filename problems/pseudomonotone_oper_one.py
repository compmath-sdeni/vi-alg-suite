import os
from typing import Union
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class PseudoMonotoneOperOne(VIProblem):
    def __init__(self,
                 *,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, vis: Sequence[VisualParams] = None,
                 hr_name: str = None, unique_name: str = 'PseudoMonotoneOperOneProblem',
                 lam_override: float = None,
                 lam_override_by_method: dict = None
                 ):
        super().__init__(xtest=np.array([0, 0, 0]), x0=x0, C=C, hr_name=hr_name, unique_name=unique_name,
                         lam_override=lam_override, lam_override_by_method=lam_override_by_method)

        self.arity = 3
        self.L = 5.068
        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = np.zeros(self.arity)

        self.M = np.array([
            [1., 0., -1.],
            [0., 1.5, 0.],
            [-1., 0., 2.]])

    def F(self, x: np.ndarray) -> float:
        # t = self.C.project(x - self.A(x))
        # return np.linalg.norm(t - x)
        # solution x = 0, so
        return np.linalg.norm(self.A(x) - x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        t: float = (np.exp(-(np.linalg.norm(x) ** 2)) + 0.2)

        return np.dot((t * self.M), x)

    def Project(self, x: np.array) -> np.array:
        return self.C.project(x)

    def Draw2DProj(self, fig, ax, vis, xdim, ydim=None, curX=None, mutableOnly=False):
        x = np.arange(vis.xl, vis.xr, (vis.xr - vis.xl) * 0.01, float)
        xv = [[t] for t in x]

        # f = np.vectorize(self.F)

        def Fcut(F, defx, u, d1):
            defx[d1] = u
            return F(defx)

        y = [Fcut(self.F, self.defaultProjection, t, xdim) for t in xv]

        res = ax.plot(x, y, 'g-')
        return res

    def Draw3DProj(self, fig, ax, vis, xdim, ydim, curX=None):
        xgrid = np.arange(vis.xl, vis.xr, (vis.xr - vis.xl) * 0.05, float)
        ygrid = np.arange(vis.yb, vis.yt, (vis.yt - vis.yb) * 0.05, float)
        mesh = [[x, y] for y in xgrid for x in ygrid]

        gridPoints = np.array([[[x, y] for y in ygrid] for x in xgrid])

        tmp = self.defaultProjection.copy()  # speed up
        zVals = np.array([self.Fcut2D(tmp, x, y, xdim, ydim) for x, y in mesh])

        # self.ax.plot_surface(X, Y, Z)
        # self.ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        # self.ax.plot_trisurf(x, y, zs, cmap=cm.jet, linewidth=0.2)
        res = ax.plot_trisurf(gridPoints[:, :, 0].flatten(), gridPoints[:, :, 1].flatten(), zVals,
                              cmap=cm.jet, linewidth=0, alpha=0.85)

        # xv = [[u, w] for u, w in zip(x,y)]
        # z = [self.stableFunc(t) for t in xv]
        # self.ax.plot_wireframe(x, y, z, rstride=10, cstride=10)

        return res

    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        np.savetxt(os.path.join(path_to_save, 'M.txt'), self.M)
        np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0)

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0)

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest)

        return path_to_save
