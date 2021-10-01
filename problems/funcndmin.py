from typing import Union
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class FuncNDMin(VIProblem):
    L = 10  # type: float

    def __init__(self,
                 arity: int, f: Callable[[np.ndarray], float], df: Callable[[np.ndarray], np.ndarray], *,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, L: float = 10, vis: Sequence[VisualParams] = None,
                 defaultProjection: np.ndarray = None,
                 xtest: Union[np.ndarray, float] = None, hr_name: str = None,
                 lam_override: float = None,
                 lam_override_by_method:dict = None
                 ):
        super().__init__(xtest=xtest, x0=x0, hr_name=hr_name, lam_override=lam_override, lam_override_by_method=lam_override_by_method)

        self.arity = arity
        self.f = f
        self.df = df
        self.C = C
        self.L = L
        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = defaultProjection if defaultProjection is not None else np.zeros(self.arity)
        if xtest is not None:
            self.xtest = xtest
        else:
            self.xtest = np.zeros(self.arity)

    def F(self, x: np.array) -> float:
        return self.f(x)

    def GradF(self, x: np.array) -> np.ndarray:
        return self.df(x)

    def A(self, x: np.array) -> np.ndarray:
        return self.GradF(x)

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
