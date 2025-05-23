import numpy as np
import networkx as nx

from problems.testcases.transport.transportation_network import TransportationNetwork


class MultiModalTransportationNetwork(TransportationNetwork):
    def __init__(self, *, edges_list, demand, modes_info, max_od_paths_count=3, max_path_edges=10):
        super().__init__(edges_list=edges_list, demand=demand,
                         max_od_paths_count=max_od_paths_count, max_path_edges=max_path_edges)
        self.modes_info = modes_info  # Dict with parameters for each mode (e.g., bike, bus, car, subway)

    def generalized_travel_cost(self, flow_vector):
        # Calculate generalized travel costs including uncertainty
        costs = np.zeros(self.graph.number_of_edges())
        for edge_key, edge_data in self.graph.edges.items():
            mode = edge_data.get('mode')
            params = self.modes_info[mode]

            # Interval travel cost calculation (simplified example)
            travel_time = params['free_flow_time'] * (1 + params['alpha'] * (flow_vector[edge_key] / params['capacity']) ** params['beta'])
            crowd_discomfort = params['crowd_discomfort'] * travel_time
            travel_fare = params['fare'] * edge_data.get('length', 1)

            # Generalized cost with weights from article
            costs[edge_key] = (params['lambda_t'] * travel_time +
                               params['lambda_u'] * crowd_discomfort +
                               params['lambda_m'] * travel_fare)
        return costs
