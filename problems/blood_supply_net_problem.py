import os
from typing import Union
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from constraints.r_plus import RnPlus
from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints

import networkx as nx


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
                 expected_shortage: Sequence[tuple], expected_surplus: Sequence[tuple], edge_loss: Sequence[float],
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

        self.expected_shortage = expected_shortage
        self.expected_surplus = expected_surplus

        self.lam_minus = lam_minus
        self.lam_plus = lam_plus
        self.theta = theta

        self.paths = paths

        self.path_loss = np.ones(len(paths))

        self.n_p = len(self.paths)
        self.n_L = len(edges)

        # list of lists of paths grouped by demand point (indices of paths)
        wk_dict = {}
        for j in range(self.n_p):
            last_edge_idx = paths[j][len(paths[j]) - 1]
            dest_node = self.edges[last_edge_idx][1]
            if dest_node not in wk_dict:
                wk_dict[dest_node] = []

            wk_dict[dest_node].append(j)

        self.wk_dict = wk_dict

        self.projected_demands = np.zeros(self.n_R)

        self.build_static_params()

        for i in range(self.n_L):
            print(f"alpha_{i}: {self.edge_loss[i]}")
            for j in range(self.n_p):
                print(f"d_{i}_{j}: {self.deltas[i, j]}, alpha_{i}_{j}: {self.alphaij[i, j]}", sep=' | ', end=' | ')

        for j in range(self.n_p):
            print(f"m_{j}: {self.path_loss[j]}")


    # koeffs and network structure information are static - need to be calculated only once
    def build_static_params(self):
        self.deltas = np.zeros((self.n_L, self.n_p))
        self.demand_points_dic = {}

        demand_point_index = 0
        for j, p in enumerate(self.paths):
            for i in p:
                self.deltas[i, j] = 1

            last_edge_idx = p[len(p) - 1]
            dest_node = self.edges[last_edge_idx][1]

            if dest_node not in self.demand_points_dic:
                self.demand_points_dic[dest_node] = demand_point_index
                demand_point_index += 1


        self.path_loss = np.ones(self.n_p)
        self.alphaij = np.copy(self.deltas)
        for j, p in enumerate(self.paths):
            mu = 1.0
            for k, i in enumerate(p):
                self.alphaij[i, j] = mu
                mu *= self.edge_loss[i]

            self.path_loss[j] = mu

    # need recalculation after every change of the paths flows x
    def recalc_link_flows_and_demands(self, x: np.ndarray):
        self.link_flows = np.zeros(self.n_L)
        self.projected_demands = np.zeros(self.n_R)

        for j in range(self.n_p):
            p = self.paths[j]
            # link flows
            for i in p:
                self.link_flows[i] += self.alphaij[i, j] * x[j]

            last_edge_idx = p[len(p) - 1]
            dest_node = self.edges[last_edge_idx][1]

            # projected demands
            self.projected_demands[self.demand_points_dic[dest_node]] += self.path_loss[j] * x[j]

    def C_hat_i(self, path_flow: float, path_index: int) -> float:
        s = 0
        for i in self.paths[path_index]:
            s += self.c[i][0](self.link_flows[i]) * self.alphaij[i, path_index]

        return path_flow * s

    def Z_hat_i(self, path_flow: float, path_index: int) -> float:
        s = 0
        for i in self.paths[path_index]:
            s += self.z[i][0](self.link_flows[i]) * self.alphaij[i, path_index]

        return path_flow * s

    def R_hat_i(self, path_flow: float, path_index: int) -> float:
        s = 0
        i = self.paths[path_index][0]
        s += self.r[i][0](self.link_flows[i]) * self.alphaij[i, path_index]

        return path_flow * s

    def E_delta_minus(self, demand_point_index: int) -> float:
        s = self.expected_shortage[demand_point_index][0](self.projected_demands[demand_point_index])
        return s

    def E_delta_plus(self, demand_point_index: int) -> float:
        s = self.expected_surplus[demand_point_index][0](self.projected_demands[demand_point_index])
        return s

    # x is a vector of path flows
    def get_loss(self, x: np.ndarray, *, recalc_link_flows: bool = False) -> float:
        if recalc_link_flows:
            self.recalc_link_flows_and_demands(x)

        oper_cost = 0
        waste_cost = 0
        risk_cost = 0

        loss = 0
        for j in range(self.n_p):
            oper_cost += self.C_hat_i(x[j], j)
            waste_cost +=  self.Z_hat_i(x[j], j)
            risk_cost += self.R_hat_i(x[j], j)

        loss += oper_cost + waste_cost + self.theta * risk_cost

        for k in range(self.n_R):
            t = self.lam_minus * self.E_delta_minus(k)
            loss += t
            t = self.lam_plus * self.E_delta_plus(k)
            loss += t

        return loss

    def get_loss_grad(self, x:np.ndarray, *, recalc_link_flows: bool = False) -> np.ndarray:
        if recalc_link_flows:
            self.recalc_link_flows_and_demands(x)

        grad = np.zeros_like(x)
        for l in range(self.n_p):
            oper_cost_diff = 0
            waste_cost_diff = 0
            risk_cost_diff = 0

            loss = 0

            for i in self.paths[l]:
                fi = self.link_flows[i]
                oper_cost_diff += (self.c[i][0](fi) + self.c[i][1](fi)*fi) * self.alphaij[i][l]

            grad[l] = oper_cost_diff

        return grad

    def to_nx_graph(self):
        G = nx.DiGraph()
        pos = dict()
        labels = dict()

        node_idx = 0
        G.add_node(node_idx, label=f"1")
        pos[node_idx] = [0.5, 60]
        labels[node_idx] = "1"
        node_idx += 1

        for i in range(self.n_C):
            G.add_node(node_idx, label=f"C{i+1}")
            pos[node_idx] = [i/self.n_C + 1/(self.n_C + 1), 50]
            labels[node_idx] = f"C{i+1}"
            node_idx += 1

        for i in range(self.n_B):
            G.add_node(node_idx, label=f"B{i+1}")
            pos[node_idx] = [i / self.n_B +  1/(self.n_B + 1) , 40]
            labels[node_idx] = f"B{i+1}"
            node_idx += 1

        for i in range(self.n_Cmp):
            G.add_node(node_idx, label=f"P{i+1}")
            pos[node_idx] = [i / self.n_Cmp + 1/(self.n_Cmp + 1), 30]
            labels[node_idx] = f"P{i+1}"
            node_idx += 1

        for i in range(self.n_S):
            G.add_node(node_idx, label=f"S{i+1}")
            pos[node_idx] = [i / self.n_S + 1/(self.n_S + 1), 20]
            labels[node_idx] = f"S{i+1}"
            node_idx += 1

        for i in range(self.n_D):
            G.add_node(node_idx, label=f"D{i+1}")
            pos[node_idx] = [i / self.n_D  + 1/(self.n_D + 1), 10]
            labels[node_idx] = f"D{i+1}"
            node_idx += 1

        for i in range(self.n_R):
            G.add_node(node_idx, label=f"R{i+1} ({self.projected_demands[i]})", proj_demand=self.projected_demands[i])
            pos[node_idx] = [i / self.n_R  + 1/(self.n_R + 1), 0]
            labels[node_idx] = f"R{i+1} ({self.projected_demands[i]})"
            node_idx += 1

        for i, e in enumerate(self.edges):
            G.add_edge(e[0], e[1], label=f"{i} ({e[0], e[1]})", weight=self.link_flows[i])

        return G, pos, labels

    def plot(self, show_flows: bool = False):
        G, pos, labels = self.to_nx_graph()

        nx.draw_networkx_nodes(G, pos, node_size=600)
        nx.draw_networkx_labels(G, pos, labels)

        nx.draw_networkx_edges(G, pos, arrows=True)

        if show_flows:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()



class BloodSupplyNetworkProblem(VIProblem):

    def __init__(self, *, network: BloodSupplyNetwork,
                 x0: Union[np.ndarray, float] = None,
                 vis: Sequence[VisualParams] = None,
                 hr_name: str = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None,
                 xtest: Union[np.ndarray, float] = None):

        super().__init__(xtest=xtest, x0=x0, C=RnPlus(network.n_p), hr_name=hr_name, lam_override=lam_override,
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
