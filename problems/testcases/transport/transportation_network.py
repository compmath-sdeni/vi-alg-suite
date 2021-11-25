from operator import itemgetter
from typing import Dict, List, Tuple, Sequence, Mapping
import networkx as nx

# Link travel time = free flow time * ( 1 + B * (flow/capacity)^Power ).
# Link generalized cost = Link travel time + toll_factor * toll + distance_factor * distance
import numpy as np


class TransportationNetwork:
    def __init__(self, *,
                 edges_list: Sequence[Tuple[int, int, Mapping]] = None,
                 demand: List[Tuple[int, int, float]] = None) -> None:

        self.paths_count = 0
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.paths: List[List[np.ndarray]] = []
        self.Q: np.ndarray = None

        if edges_list:
            # sorted_edges = sorted(edges_list, key=itemgetter(0, 1))

            k: int = 0  # edge key = zero based edge number for algs
            for e in edges_list:
                self.graph.add_edge(e[0], e[1], key=k, **e[2])
                k += 1

            # self.graph.add_edges_from(sorted_edges)

        if demand:
            self.demand: List[Tuple[int, int, float]] = demand
        else:
            self.demand: List[Tuple[int, int, float]] = []

        if edges_list and demand:
            self.calc_paths()
            self.Q = self._calc_edges_to_paths_incidence_()

    def show(self):
        print(self.graph)
        print(self.graph.edges(data=True))
        print("Demand: ")
        for d in self.demand:
            print(f'{d[0]} -> {d[1]}: {d[2]}')

        print("Paths: ")
        print(self.paths)

    def draw(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, labels={node: node for node in self.graph.nodes()})
        nx.draw_networkx_edge_labels(self.graph, pos)

    def calc_paths(self):
        self.paths = []
        self.paths_count = 0
        for d in self.demand:
            paths_for_pair: List = []
            for p in nx.all_simple_edge_paths(self.graph, d[0], d[1]):
                edge_keys = np.ndarray((len(p),), dtype=int)
                i = 0
                for e in p:
                    edge_keys[i] = e[2]
                    i += 1
                paths_for_pair.append(edge_keys)
                self.paths_count += 1

            self.paths.append(paths_for_pair)

    def get_demands_vector(self) -> np.ndarray:
        return np.array([d[2] for d in self.demand])

    def get_paths_to_demands_incidence(self) -> List[np.ndarray]:
        res: List[np.ndarray] = []
        i = 0
        for d, paths_for_pair in zip(self.demand, self.paths):
            res.append(np.array(range(i, len(paths_for_pair))))
            i += len(paths_for_pair)

        return res

    def _calc_edges_to_paths_incidence_(self) -> np.ndarray:
        res: np.ndarray = np.zeros((self.paths_count, len(self.graph.edges)), dtype=float)
        i = 0
        for paths_for_pair in self.paths:
            for p in paths_for_pair:
                res[i][p] = 1 # p is np.ndarray of edge keys (zero-based sequential edge indices) for the path
                i += 1

        return res.T

    def get_cost_function(self):
        N: int = self.graph.number_of_edges()
        free_flows = np.ndarray((N,), dtype=float)
        koeffs = np.ndarray((N,), dtype=float)
        capacity_inv = np.ndarray((N,), dtype=float)
        power = np.ndarray((N,), dtype=float)
        for i, e in enumerate(self.graph.edges.data()):
            free_flows[i] = e[2]['frf']
            koeffs[i] = e[2]['k']
            capacity_inv[i] = 1./e[2]['cap']
            power[i] = e[2]['pow']

        def cost(x: np.ndarray) -> np.ndarray:
            return self.Q.T @ (free_flows * (1.0 + koeffs * np.power(((self.Q @ x) * capacity_inv), power)))

        return cost
