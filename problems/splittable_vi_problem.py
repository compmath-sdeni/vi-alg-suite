from typing import Union, List
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class SplittableVIProblem(VIProblem):
    def __init__(self, M: int, *, K: int = None,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints, vis: Sequence[VisualParams] = None,
                 defaultProjection: np.ndarray = None, xtest: Union[np.ndarray, float] = None, hr_name: str = None, lam_override: float = None):
        super().__init__(xtest=xtest, x0=x0, C=C, hr_name=hr_name, lam_override=lam_override)

        self.blocks: np.ndarray = None

        self.M: int = M
        self.K: int = 1

        self.set_split(K)

        self.f: List[Callable[[np.ndarray], float]] = None
        self.df: List[Callable[[np.ndarray], np.ndarray]] = None

        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = defaultProjection if defaultProjection is not None else np.zeros(self.M)
        if xtest is not None:
            self.xtest = xtest
        else:
            self.xtest = np.zeros(self.M)

    def set_split(self, new_k: int):
        self.K = new_k if new_k is not None else 1
        self.blocks = np.empty(self.K, dtype=int)

        part_len: int = self.M // self.K
        last_len = self.M - part_len * (self.K - 1)

        if self.K > 1:
            self.blocks[:self.K - 1] = part_len
            self.blocks[self.K - 1] = part_len if last_len == 0 else last_len
        else:
            self.blocks[0] = self.M

    def F(self, x: np.array) -> float:
        return sum([f(x) for f in self.f])

    def GradF(self, x: np.array) -> np.ndarray:
        return np.sum([df(x) for df in self.df], 0)

    def Fi(self, x: np.array, i: int) -> float:
        # return self.f[i](x)
        return np.sum([f(x) for f in self.f[i*self.K:i*self.K+self.blocks[i]]])

    def GradFi(self, x: np.array, i: int) -> float:
        # t = self.GradF(x)
        # t = self.df[i](x)
        t = np.sum([df(x) for df in self.df[i * self.blocks[0]:i * self.blocks[0] + self.blocks[i]]], 0)
        return t

    def Project(self, x: np.array) -> np.array:
        return self.C.project(x) if self.C is not None else x

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

        grid_points = np.array([[[x, y] for y in ygrid] for x in xgrid])

        tmp = self.defaultProjection.copy()  # speed up
        z_vals = np.array([self.Fcut2D(tmp, x, y, xdim, ydim) for x, y in mesh])

        # self.ax.plot_surface(X, Y, Z)
        # self.ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        # self.ax.plot_trisurf(x, y, zs, cmap=cm.jet, linewidth=0.2)
        res = ax.plot_trisurf(grid_points[:, :, 0].flatten(), grid_points[:, :, 1].flatten(), z_vals,
                              cmap=cm.jet, linewidth=0, alpha=0.85)

        # xv = [[u, w] for u, w in zip(x,y)]
        # z = [self.stableFunc(t) for t in xv]
        # self.ax.plot_wireframe(x, y, z, rstride=10, cstride=10)

        return res
