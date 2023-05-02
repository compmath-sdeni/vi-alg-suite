import os
from typing import Union
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


# Network is defined by the next data, which is passed to the constructor
# * Number of collection sites n_C, blood centers n_B, component labs n_P, storage facilities n_S, distribution centers n_D and demand points n_R
# * List of edges, where each edge is a tuple of node indices: (from, to);
# * Also passed list of tuples with the same indexing as the edge list: operation unit cost functions (c), waste unit cost functions (z),
# * risk unit cost functions (r) (used only for the first layer of edges). All functions are passed as a tuple of function and its derivative.
# * List of shortage and surplus expectations with their derivatives (E \delta^-, E \delta^+, E' \delta^-, E' \delta^+)
# * Shortage penalty lambda_-, surplus penalty lambda_+ and risk weight \theta
# Optionally precomputed paths can be passed as well, where each path is a list indices from edge list.

class BloodSupplyNetwork:
    def __init__(self, *, n_C: int, n_B: int, n_Cmp: int, n_S: int, n_D: int, n_R: int,
                 edges: Sequence[tuple], c: Sequence[tuple], z: Sequence[tuple], r: Sequence[tuple],
                 shortage: Sequence[tuple], surplus: Sequence[tuple], edge_loss: Sequence[float],
                 lam_minus: float, lam_plus: float, theta: float, paths: Sequence[Sequence[int]] = None
                 ):
        self.nodes_count = n_C + n_B + n_Cmp + n_S + n_D + n_R
        self.n_C = n_C
        self.n_B = n_B
        self.n_Cmp = n_Cmp
        self.n_S = n_S
        self.n_D = n_D
        self.n_R = n_R

        self.edges = edges

        self.c = c
        self.z = z
        self.r = r

        self.edge_loss = edge_loss

        self.shortage = shortage
        self.surplus = surplus

        self.lam_minus = lam_minus
        self.lam_plus = lam_plus
        self.theta = theta

        self.paths = paths

        self.path_loss = np.ones(len(paths))

        self.n_p = len(self.paths)
        self.n_L = len(edges)

        # list of lists of paths grouped by demand point (indices of paths)
        wk_list = [[] for i in range(self.n_p)]
        for j in range(self.n_p):
            last_edge_idx = paths[j][len(paths[j]) - 1]
            wk_list[self.edges[last_edge_idx][1]].append(j)

        self.wk_list = wk_list

        self.build_koeffs()

    def build_koeffs(self):
        self.deltas = np.zeros((self.n_L, self.n_p))
        for j, p in enumerate(self.paths):
            for i in p:
                self.deltas[i, j] = 1

        self.path_loss = np.ones(self.n_p)
        self.alphaij = np.copy(self.deltas)
        for j, p in enumerate(self.paths):
            mu = 1.0
            for k, i in enumerate(p):
                self.alphaij[i, j] = mu
                mu *= self.edge_loss[i]

            self.path_loss[j] = mu
    def C_hat_i(self, x: np.ndarray, path_index: int) -> float:
        return self.c[0] * x[0]


class BloodSupplyNetworkProblem(VIProblem):

    def __init__(self, *, network: BloodSupplyNetwork,
                 x0: Union[np.ndarray, float] = None,
                 vis: Sequence[VisualParams] = None,
                 hr_name: str = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None,
                 xtest: Union[np.ndarray, float] = None):

        super().__init__(xtest=xtest, x0=x0, C=C, hr_name=hr_name, lam_override=lam_override,
                         lam_override_by_method=lam_override_by_method)

        self.net = network
        self.arity = self.net.n_p

        self.vis = vis if vis is not None else VisualParams()
        self.defaultProjection = np.zeros(self.arity)

    def F(self, x: np.ndarray) -> float:
        e = 2.5 - x if x < 2.5 else 0
        return 11 * x[0] ** 2 + 38 * x + 100 * e

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        e = 0 if x[0] <= 0 else (1 if x[0] >= 5 else x[0] / 5)
        v = 22 * x[0] + 100 * (e - 1) + 38
        return np.array([v])

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

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0)

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest)

        return path_to_save
