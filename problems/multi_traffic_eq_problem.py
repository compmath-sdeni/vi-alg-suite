from typing import Union

import numpy as np

from methods.projections.simplex_projection_prom import vec2simplexV2
from problems.testcases.transport.multi_transport_net import MultiModalTransportationNetwork
from constraints.allspace import Rn
from problems.traffic_equilibrium import TrafficEquilibrium


class MultiModalTrafficEquilibriumProblem(TrafficEquilibrium):
    def __init__(self,
                 multimodal_network: MultiModalTransportationNetwork,
                 demands_interval,
                 x0: Union[np.ndarray] = None):
        self.multimodal_network = multimodal_network
        self.demands_interval = demands_interval  # Demand intervals instead of fixed numbers
        d = np.mean(demands_interval, axis=1)  # Using mean as representative initial demand
        W = multimodal_network.get_paths_to_demands_incidence()
        Q = multimodal_network.Q
        n = Q.shape[1]

        super().__init__(Gf=self.calculate_operator,
                         d=d, W=W, Q=Q, network=multimodal_network,
                         x0=x0,
                         hr_name="MultiModalTrafficEquilibrium",
                         C=Rn(n))

    def calculate_operator(self, x):
        # Operator A: generalized travel costs
        return self.multimodal_network.generalized_travel_cost(self.Q @ x)

    def Project(self, x):
        # Projecting the path flows to feasible set considering demands interval
        res = x.copy()
        for i, demand_paths in enumerate(self.W):
            interval = self.demands_interval[i]
            mean_demand = np.mean(interval)
            proj_part = vec2simplexV2(res[demand_paths], mean_demand)
            res[demand_paths] = proj_part
        return res
