import json
import os
from typing import Union, Dict, Optional
from typing import Callable, Sequence

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from constraints.r_plus import RnPlus
from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints

import pickle

import networkx as nx


# Network is defined by the next data, which is passed to the constructor
# * Number of collection sites n_C, blood centers n_B, component labs n_P, storage facilities n_S, distribution centers n_D and demand points n_R
# * List of edges, where each edge is a tuple of node indices: (from, to);
# * Also passed list of tuples with the same indexing as the edge list: operation unit cost functions (c), waste unit cost functions (z),
# * risk unit cost functions (r) (used only for the first layer of edges). All functions are passed as a tuple of function and its derivative.
# * List of shortage and surplus expectations with their derivatives (E \delta^-, E \delta^+, E' \delta^-, E' \delta^+)
# * Shortage penalty lambda_-, surplus penalty lambda_+ and risk weight \theta
# Optionally precomputed paths can be passed as well, where each path is a list indices from edge list.

# https://machinelearningmastery.com/calculating-derivatives-in-pytorch/
class BloodSupplyNetwork:
    def __init__(self, *, n_C: int, n_B: int, n_Cmp: int, n_S: int, n_D: int, n_R: int,
                 edges: Sequence[tuple], c: Sequence[tuple], z: Sequence[tuple], r: Sequence[tuple],
                 expected_shortage: Sequence[tuple], expected_surplus: Sequence[tuple], edge_loss: Sequence[float],
                 lam_minus: Sequence[float], lam_plus: Sequence[float], theta: float,
                 paths: Sequence[Sequence[int]] = None
                 ):
        self.nodes_count = 1 + n_C + n_B + n_Cmp + n_S + n_D + n_R  # also we have virtual node 0 - source node, "regional division"
        self.n_C = n_C
        self.n_B = n_B
        self.n_Cmp = n_Cmp
        self.n_S = n_S
        self.n_D = n_D
        self.n_R = n_R

        self.edges = edges

        self.adj_dict = {}
        for idx, e in enumerate(self.edges):
            if e[0] not in self.adj_dict:
                self.adj_dict[e[0]] = {e[1]: idx}
            else:
                self.adj_dict[e[0]][e[1]] = idx

        self.c = c
        self.z = z
        self.r = r

        self.edge_loss = edge_loss

        self.expected_shortage = expected_shortage
        self.expected_surplus = expected_surplus

        self.lam_minus = lam_minus
        self.lam_plus = lam_plus
        self.theta = theta

        self.n_L = len(edges)

        self.projected_demands = np.zeros(self.n_R)

        self.build_demand_points_dict()

        self.G, self.pos, self.labels = self.to_nx_graph()

        if paths is None:
            demand_points = [k for k in self.demand_points_dic.keys()]
            self.paths = []
            for path in nx.all_simple_paths(self.G, source=0, target=demand_points):
                path_edge_indices = []
                v1 = path[0]
                for i in range(1, len(path)):
                    v2 = path[i]
                    path_edge_indices.append(self.adj_dict[v1][v2])
                    v1 = v2
                self.paths.append(path_edge_indices)
        else:
            self.paths = paths

        self.path_loss = np.ones(len(self.paths))
        self.n_p = len(self.paths)

        self.build_static_params()

    # sanity check function - calculate projected demands (final supplies) by edge flows, edge loss coeffs,
    # edge operational costs, risks and expectations
    def sanity_check(self):
        v = [6.06, 44.05, 30.99]
        f = [54.72, 43.90, 30.13, 22.42, 19.57, 23.46, 49.39, 42.00, 43.63, 39.51, 29.68, 13.08, 26.20, 13.31, 5.78,
             25.78, 24.32, .29, 18.28, 7.29]

        edge_loss = [0.97, 0.99, 1.00, 0.99, 1.00, 1.00, 0.92, 0.96, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
                     0.98, 1.00, 1.00, 0.98]
        total_edge_oper_cost = [lambda t: 6 * t ** 2 + 15 * t, lambda t: 9 * t ** 2 + 11 * t,
                                lambda t: 0.7 * t ** 2 + t, lambda t: 1.2 * t ** 2 + t,
                                lambda t: 1 * t ** 2 + 3 * t, lambda t: 0.8 * t ** 2 + 2 * t,
                                lambda t: 2.5 * t ** 2 + 2 * t, lambda t: 3 * t ** 2 + 5 * t,
                                lambda t: 0.8 * t ** 2 + 6 * t, lambda t: 0.5 * t ** 2 + 3 * t,
                                lambda t: 0.3 * t ** 2 + t, lambda t: 0.5 * t ** 2 + 2 * t,
                                lambda t: 0.4 * t ** 2 + 2 * t, lambda t: 0.6 * t ** 2 + t,
                                lambda t: 1.3 * t ** 2 + 3 * t, lambda t: 0.8 * t ** 2 + 2 * t,
                                lambda t: 0.5 * t ** 2 + 3 * t, lambda t: 0.7 * t ** 2 + 2 * t,
                                lambda t: 0.6 * t ** 2 + 4 * t, lambda t: 1.1 * t ** 2 + 5 * t]
        total_edge_waste_cost = [lambda t: 0.8 * t ** 2, lambda t: 0.7 * t ** 2, lambda t: 0.6 * t ** 2,
                                 lambda t: 0.8 * t ** 2, lambda t: 0.6 * t ** 2, lambda t: 0.8 * t ** 2,
                                 lambda t: 0.5 * t ** 2, lambda t: 0.8 * t ** 2, lambda t: 0.4 * t ** 2,
                                 lambda t: 0.7 * t ** 2, lambda t: 0.3 * t ** 2, lambda t: 0.4 * t ** 2,
                                 lambda t: 0.3 * t ** 2, lambda t: 0.4 * t ** 2, lambda t: 0.7 * t ** 2,
                                 lambda t: 0.4 * t ** 2, lambda t: 0.5 * t ** 2, lambda t: 0.7 * t ** 2,
                                 lambda t: 0.4 * t ** 2, lambda t: 0.5 * t ** 2]

        nodes_in_flow = np.zeros(self.nodes_count)
        nodes_out_flow = np.zeros(self.nodes_count)

        for i, e in enumerate(self.edges):
            nodes_in_flow[e[1]] += f[i] * edge_loss[i]
            nodes_out_flow[e[0]] += f[i]

        print(f"In-flow: {nodes_in_flow}")
        print(f"Out-flow: {nodes_out_flow}")
        print(f"Divergency: {nodes_in_flow - nodes_out_flow}")
        print("Looks like data is not fully consistent!!!")

    # dictionary of demand_point node id (integer, number of the node in network) ->
    # demand_point_index (zero-based index of the demand point)
    # actually with current node indexing, can be replaced with a simple shift formula
    def build_demand_points_dict(self):
        nodes_cnt = self.nodes_count
        current_demand_node = nodes_cnt - self.n_R

        self.demand_points_dic = {}
        for i in range(self.n_R):
            self.demand_points_dic[current_demand_node + i] = i

        self.last_layer_links = []
        for i, e in enumerate(self.edges):
            if e[1] in self.demand_points_dic:
                self.last_layer_links.append(i)

    # koeffs and network structure information are static - need to be calculated only once
    def build_static_params(self):
        self.deltas = np.zeros((self.n_L, self.n_p))

        # list of demand_point_index -> list of paths to that demand point
        self.wk_list = [[] for _ in range(self.n_R)]
        for j in range(self.n_p):
            last_edge_idx = self.paths[j][len(self.paths[j]) - 1]
            dest_node = self.edges[last_edge_idx][1]
            demand_point_index = self.demand_points_dic[dest_node]
            self.wk_list[demand_point_index].append(j)

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
            pflow = x[j]
            # link flows
            for i in p:
                self.link_flows[i] += pflow
                pflow *= self.edge_loss[i]
                # self.link_flows[i] += self.alphaij[i, j] * x[j]

            self.projected_demands[self.demand_points_dic[self.edges[i][1]]] += pflow

            # last_edge_idx = p[len(p) - 1]
            # dest_node = self.edges[last_edge_idx][1]
            #
            # # projected demands
            # self.projected_demands[self.demand_points_dic[dest_node]] += self.path_loss[j] * x[j]

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
            waste_cost += self.Z_hat_i(x[j], j)
            risk_cost += self.R_hat_i(x[j], j)

        loss += oper_cost + waste_cost + self.theta * risk_cost

        for k in range(self.n_R):
            t = self.lam_minus[k] * self.E_delta_minus(k)
            loss += t
            t = self.lam_plus[k] * self.E_delta_plus(k)
            loss += t

        return loss

    def get_loss_by_link_flows(self) -> float:
        oper_cost = 0
        waste_cost = 0
        risk_cost = 0

        loss = 0
        for i in range(self.n_L):
            fi = self.link_flows[i]
            oper_cost += fi * self.c[i][0](fi)

            waste_cost += fi * self.z[i][0](fi)

        for i in range(self.n_C):
            fi = self.link_flows[i]
            risk_cost += fi * self.r[i][0](fi)

        loss += oper_cost + waste_cost + self.theta * risk_cost

        for k in range(self.n_R):
            t = self.lam_minus[k] * self.E_delta_minus(k)
            loss += t
            t = self.lam_plus[k] * self.E_delta_plus(k)
            loss += t

        return loss

    def get_demands_by_link_flows(self) -> Sequence[float]:
        v = [0 for _ in range(self.n_R)]
        for i in self.last_layer_links:
            v[self.demand_points_dic[self.edges[i][1]]] += self.link_flows[i] * self.edge_loss[i]
        return v

    def get_loss_grad(self, x: np.ndarray, *, recalc_link_flows: bool = False) -> np.ndarray:
        if recalc_link_flows:
            self.recalc_link_flows_and_demands(x)

        grad = np.zeros_like(x)
        for l in range(self.n_p):
            oper_cost_diff = 0
            waste_cost_diff = 0

            for i in self.paths[l]:
                fi = self.link_flows[i]
                oper_cost_diff += (self.c[i][0](fi) + self.c[i][1](fi) * fi) * self.alphaij[i][l]
                waste_cost_diff += (self.z[i][0](fi) + self.z[i][1](fi) * fi) * self.alphaij[i][l]

            i_star = self.paths[l][0]
            fi_star = self.link_flows[i_star]
            risk_cost_diff = self.theta * (self.r[i_star][0](fi_star) + self.r[i_star][1](
                fi_star) * fi_star)  # not needed! *self.alphaij[i, l]

            last_edge_idx = self.paths[l][len(self.paths[l]) - 1]
            dest_node = self.edges[last_edge_idx][1]
            demand_point_index = self.demand_points_dic[dest_node]

            v_k_l = self.projected_demands[demand_point_index]
            s = self.expected_shortage[demand_point_index][1](v_k_l)
            shortage_cost_diff = self.lam_minus[demand_point_index] * s * self.path_loss[l]

            s = self.expected_surplus[demand_point_index][1](v_k_l)
            surplus_cost_diff = self.lam_plus[demand_point_index] * s * self.path_loss[l]

            grad[l] = oper_cost_diff + waste_cost_diff + risk_cost_diff + shortage_cost_diff + surplus_cost_diff

        return grad

    def to_nx_graph(self, *, use_flows_and_demands: bool = False, x_left=0, x_right=1, y_bottom=0, y_top=60):
        G = nx.DiGraph()
        pos = dict()
        labels = dict()

        w = x_right - x_left
        h = y_top - y_bottom
        h_step = h / 6
        row_y = y_top

        node_idx = 0
        G.add_node(node_idx, label=f"1")
        pos[node_idx] = [(x_left+x_right) / 2 + 0.08*w, row_y]
        labels[node_idx] = "1"
        node_idx += 1

        row_y -= h_step

        for i in range(self.n_C):
            G.add_node(node_idx, label=f"C{i + 1}")
            pos[node_idx] = [x_left + (i / self.n_C + 1 / (self.n_C + 1))*w, row_y]
            labels[node_idx] = f"C{i + 1}"
            node_idx += 1

        row_y -= h_step
        for i in range(self.n_B):
            G.add_node(node_idx, label=f"B{i + 1}")
            pos[node_idx] = [x_left + (i / self.n_B + 1 / (self.n_B + 1))*w, row_y]
            labels[node_idx] = f"B{i + 1}"
            node_idx += 1

        row_y -= h_step
        for i in range(self.n_Cmp):
            G.add_node(node_idx, label=f"P{i + 1}")
            pos[node_idx] = [x_left + (i / self.n_Cmp + 1 / (self.n_Cmp + 1))*w, row_y]
            labels[node_idx] = f"P{i + 1}"
            node_idx += 1

        row_y -= h_step
        for i in range(self.n_S):
            G.add_node(node_idx, label=f"S{i + 1}")
            pos[node_idx] = [x_left + (i / self.n_S + 1 / (self.n_S + 1))*w, row_y]
            labels[node_idx] = f"S{i + 1}"
            node_idx += 1

        row_y -= h_step
        for i in range(self.n_D):
            G.add_node(node_idx, label=f"D{i + 1}")
            pos[node_idx] = [x_left + (i / self.n_D + 1 / (self.n_D + 1))*w, row_y]
            labels[node_idx] = f"D{i + 1}"
            node_idx += 1

        row_y -= h_step
        for i in range(self.n_R):
            G.add_node(node_idx, label=f"R{i + 1} ({self.projected_demands[i]})", proj_demand=self.projected_demands[i])
            pos[node_idx] = [x_left + (i / self.n_R + 1 / (self.n_R + 1))*w, row_y]
            labels[node_idx] = f"R{i + 1} ({self.projected_demands[i]:.2f})"
            node_idx += 1

        for i, e in enumerate(self.edges):
            if use_flows_and_demands:
                G.add_edge(e[0], e[1], label=f"{i} ({e[0], e[1]})", weight=f"{self.link_flows[i]:.2f}",
                           edge_index=f"{i + 1}")
            else:
                G.add_edge(e[0], e[1], label=f"{i} ({e[0], e[1]})", weight=f"{0}",
                           edge_index=f"{i + 1}")

        return G, pos, labels

    def plot(self, show_flows: bool = False):
        nx.draw_networkx_nodes(self.G, self.pos, node_size=300)
        nx.draw_networkx_labels(self.G, self.pos, self.labels)

        nx.draw_networkx_edges(self.G, self.pos, arrows=True)

        if show_flows:
            edge_labels = nx.get_edge_attributes(self.G, 'edge_index')
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, label_pos=0.7, font_size=12)

        plt.show()

    def saveToDir(self, *, path_to_save: str):
        # save network to pickle binary
        # with open(f"{path_to_save}/network.pickle", "wb") as file:
        #     pickle.dump(self, file)

        with open(f"{path_to_save}/network.json", "w") as file:
            network_data = {
                "n_C": self.n_C,
                "n_B": self.n_B,
                "n_Cmp": self.n_Cmp,
                "n_S": self.n_S,
                "n_D": self.n_D,
                "n_R": self.n_R,
                "n_p": self.n_p,
            }

            json.dump(network_data, file)


    @staticmethod
    def loadFromDir(*, path_to_load: str):
        with open(f"{path_to_load}/network.pickle", "rb") as file:
            return pickle.load(file)


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
        return self.net.get_loss(x)

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        self.net.recalc_link_flows_and_demands(x)
        return self.net.get_loss_grad(x)

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

        if self.net is not None:
            self.net.saveToDir(path_to_save=path_to_save)

        return path_to_save

    def GetExtraIndicators(self, x: Union[np.ndarray, float], *, averaged_x: np.ndarray = None, final: bool = False) -> \
    Optional[Dict]:
        return {'v': self.net.projected_demands, 'demand_by_link_flows': self.net.get_demands_by_link_flows(),
                'f': self.net.link_flows}
