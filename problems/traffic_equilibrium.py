import os
import sys
from typing import Union, Dict, Optional, Callable, List

import numpy as np

from constraints.classic_simplex import ClassicSimplex
from methods.projections.simplex_projection_prom import vec2simplexV2
from problems.testcases.transport.transportation_network import TransportationNetwork
from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class TrafficEquilibrium(VIProblem):
    def __init__(self, *,
                 Gf: Callable[[np.ndarray], np.ndarray] = None,  # Cost function for paths, dependent on traffic
                 d: np.ndarray = None,  # demands for source-destination pairs
                 W: List[np.ndarray] = None,  # paths to demands incidence matrix
                 Q: np.ndarray = None,  # paths to edges correspondence matrix
                 network: TransportationNetwork = None,
                 x0: Union[np.ndarray] = None,
                 C: ConvexSetConstraints,
                 hr_name: str = None,
                 x_test: np.ndarray = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None,
                 flow_eps: float = 0.0000001,
                 zero_cutoff: float = None,
                 auto_update_structure: bool = False,
                 structure_update_freq: int = 1
                 ):
        super().__init__(xtest=x_test, x0=x0, C=C, hr_name=hr_name, lam_override=lam_override,
                         lam_override_by_method=lam_override_by_method, zero_cutoff=zero_cutoff,
                         auto_update_structure=auto_update_structure, structure_update_freq=structure_update_freq)

        self.network = network

        if network is not None:
            self.d = network.get_demands_vector()
            self.Q = network.Q
            self.Gf = network.get_cost_function()
            self.W = network.get_paths_to_demands_incidence()
            self.n = network.Q.shape[1]
        else:
            self.Gf = Gf
            self.d = d
            self.W = W
            self.Q = Q
            self.n: int = len(W)  # paths count

        self.flow_eps = flow_eps

    def F(self, x: np.ndarray) -> float:
        # GAP

        costs = self.Gf(x)
        g = np.ndarray(self.d.shape)

        k = 0
        for demand_paths in self.W:
            g[k] = np.max(costs[demand_paths[x[demand_paths] > self.flow_eps]]) - np.min(
                costs[demand_paths[x[demand_paths] > self.flow_eps]])

            k += 1

        return np.max(g)  # priciest path expenses vs cheapest path expenses

    def GradF(self, x: np.ndarray) -> np.ndarray:
        return self.A(x)

    def A(self, x: np.ndarray) -> np.ndarray:
        return self.Gf(x)

    def Project(self, x: np.array) -> np.array:
        res = x.copy()
        for i in range(self.d.shape[0]):
            t = res[self.W[i]]
            proj_part = vec2simplexV2(t, self.d[i])
            res[self.W[i]] = proj_part

        return res

    def updateStructure(self, x: Union[np.ndarray, float]) -> bool:
        traffic_on_edges = self.Q @ x
        self.network.calc_paths(traffic_on_edges=traffic_on_edges)

        self.Q = self.network.Q
        self.W = self.network.get_paths_to_demands_incidence()
        self.Gf = self.network.get_cost_function()
        self.n = self.network.Q.shape[1]

    def getIndividualLoss(self, x: np.ndarray) -> float:
        return self.Gf(x).max()

    def getFlowsBetweenODPairs(self, x: np.ndarray) -> np.ndarray:
        res = np.ndarray(self.d.shape)
        k = 0
        for demand_paths in self.W:
            res[k] = np.sum(x[demand_paths])
            k += 1

        return res


    def saveToDir(self, *, path_to_save: str = None):
        path_to_save = super().saveToDir(path_to_save=path_to_save)

        # TODO!
        # np.savetxt("{0}/{1}".format(path_to_save, 'W.txt'), self.W, delimiter=',', newline="],\n[")
        np.savetxt("{0}/{1}".format(path_to_save, 'd.txt'), self.d, delimiter=',', newline="],\n[")

        if self.xtest is not None:
            np.savetxt("{0}/{1}".format(path_to_save, 'x_test.txt'), self.xtest, delimiter=',')

        if self.x0 is not None:
            np.savetxt(os.path.join(path_to_save, 'x0.txt'), self.x0, delimiter=',')

        return path_to_save

    def loadFromFile(self, path: str):
        # TODO!
        # self.W = np.loadtxt("{0}/{1}".format(path, 'W.txt'))
        self.d = np.loadtxt("{0}/{1}".format(path, 'd.txt'))
        self.xtest = np.loadtxt("{0}/{1}".format(path, 'x_test.txt'))

    def GetExtraIndicators(self, x: Union[np.ndarray, float], *, averaged_x: np.ndarray = None, final: bool = False) -> \
    Optional[Dict]:

        od_flows: np.ndarray = self.getFlowsBetweenODPairs(x)
        res = {
            "Individual loss": self.getIndividualLoss(x),
            "Cost from final flow": self.Gf(x),
            "Flow on edges": self.Q @ x,
            "Distance to feasible set": np.linalg.norm(x - self.Project(x)),
            "Flows between OD pairs": od_flows,
            "Flows between OD pairs delta": self.d - od_flows,
            "Flows error (abs, rel)":((self.d - od_flows).sum(), (self.d - od_flows).sum()/self.d.sum()),
            "Gap": self.F(x)
        }

        return res
