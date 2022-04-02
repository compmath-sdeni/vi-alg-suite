import os
import sys
from collections import Iterable
from enum import Enum, unique

import pandas as pd
from typing import Dict, List, Tuple, Sequence, Mapping
import networkx as nx

# Link travel time = free flow time * ( 1 + B * (flow/capacity)^Power ).
# Link generalized cost = Link travel time + toll_factor * toll + distance_factor * distance
import numpy as np


@unique
class EdgeParams(Enum):
    FRF = 'free_flow_time',
    CAP = 'capacity',
    K = 'bpr_coeff',
    POW = 'bpr_power',
    LEN = 'length'

    def __str__(self):
        return str.lower(self.name)


class TransportationNetwork:
    def __init__(self, *,
                 network_graph_file: str = None,
                 demands_file: str = None,
                 edges_list: Sequence[Tuple[int, int, Mapping]] = None,
                 demand: List[Tuple[int, int, float]] = None) -> None:

        self.paths_count = 0
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.paths: List[List[np.ndarray]] = []
        self.Q: np.ndarray = None
        self.total_demand: float = 0

        if network_graph_file:
            self.load_network_graph(network_graph_file)

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
            self.recalc_derived_data()

    def show(self):
        print(self.graph)
        print(list(self.graph.edges(data=True))[:5])
        print("Demand: ")
        for d in self.demand[:5]:
            print(f'{d[0]} -> {d[1]}: {d[2]}')

        print("Paths: ")
        print(self.paths[:5])

    def draw(self):
        pos = nx.planar_layout(self.graph)
        nx.draw(self.graph, pos, labels={node: node for node in self.graph.nodes()})
        nx.draw_networkx_edge_labels(self.graph, pos)

    def recalc_derived_data(self):
        self.calc_paths()
        self.Q = self._calc_edges_to_paths_incidence_()

    def estimate_path_cost(self, path: Iterable):
        cost: float = 0.
        for e in path:
            edge_params = self.graph[e[0]][e[1]][e[2]]

            # edge_params: dict = .data
            if str(EdgeParams.LEN) in edge_params:
                cost += edge_params[str(EdgeParams.LEN)]
            elif edge_params[str(EdgeParams.FRF)] >= 1:
                cost += edge_params[str(EdgeParams.FRF)]
            else:
                cost += 1
        return cost


    def calc_paths(self, max_count: int = 3, max_depth: int = 10):
        self.paths = []
        self.paths_count = 0
        for d in self.demand:
            paths_for_pair_edge_ids: List = []
            paths = list(nx.all_simple_edge_paths(self.graph, d[0], d[1], cutoff=max_depth))
            if len(paths) > max_count:
                cost_ordered_paths = sorted(paths, key=self.estimate_path_cost)
                best_paths = cost_ordered_paths[:max_count]
            else:
                best_paths = paths

            for p in best_paths:
                edge_keys = np.ndarray((len(p),), dtype=int)
                i = 0
                for e in p:
                    edge_keys[i] = e[2]
                    i += 1
                paths_for_pair_edge_ids.append(edge_keys)
                self.paths_count += 1

            self.paths.append(paths_for_pair_edge_ids)

    def get_demands_vector(self) -> np.ndarray:
        return np.array([d[2] for d in self.demand])

    def get_paths_to_demands_incidence(self) -> List[np.ndarray]:
        res: List[np.ndarray] = []
        i = 0
        for d, paths_for_pair in zip(self.demand, self.paths):
            res.append(np.array(range(i, i + len(paths_for_pair))))
            i += len(paths_for_pair)

        return res

    def _calc_edges_to_paths_incidence_(self) -> np.ndarray:
        res: np.ndarray = np.zeros((self.paths_count, len(self.graph.edges)), dtype=float)
        i = 0
        for paths_for_pair in self.paths:
            for p in paths_for_pair:
                res[i][p] = 1  # p is np.ndarray of edge keys (zero-based sequential edge indices) for the path
                i += 1

        return res.T

    def load_demands(self, file_path: str):
        f = open(file_path, 'r')
        all_rows = f.read()
        blocks = all_rows.split('Origin')[1:]
        matrix = {}

        self.total_demand = 0.
        self.demand = []

        for k in range(len(blocks)):
            orig = blocks[k].split('\n')
            dests = orig[1:]
            orig = int(orig[0])

            d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
            destinations = {}
            for i in d:
                destinations = {**destinations, **i}

            for k in destinations:
                if k != orig and  destinations[k] > 0:
                    self.demand.append((orig, k, destinations[k]))
                    self.total_demand += destinations[k]

            matrix[orig] = destinations
        zones = max(matrix.keys())
        mat: np.ndarray = np.zeros((zones, zones))
        for i in range(zones):
            for j in range(zones):
                # We map values to a index i-1, as Numpy is base 0
                mat[i, j] = matrix.get(i + 1, {}).get(j + 1, 0)

    def load_network_graph(self, edges_list_file_path: str, demands_file_path: str):
        net: pd.DataFrame = pd.read_csv(edges_list_file_path, skiprows=8, sep='\t')

        trimmed = [s.strip().lower() for s in net.columns]
        net.columns = trimmed

        # And drop the silly first and last columns
        net.drop(['~', ';'], axis=1, inplace=True)

        # net['link_type'] = net['link_type'].map(lambda x: x.rstrip(' ;'))

        net.rename(columns={'free_flow_time': str(EdgeParams.FRF), 'capacity': str(EdgeParams.CAP),
                            'b': str(EdgeParams.K), 'power': str(EdgeParams.POW), 'length': str(EdgeParams.LEN)},
                   inplace=True)

        net['edge_id'] = np.arange(len(net))

        # show for debug
        # print(net.head())
        # net.to_csv('test.csv', sep='\t')

        self.graph = nx.from_pandas_edgelist(net, 'init_node', 'term_node',
                                [str(EdgeParams.FRF), str(EdgeParams.CAP), str(EdgeParams.K),
                                 str(EdgeParams.POW), str(EdgeParams.LEN)],
                                create_using=nx.MultiDiGraph, edge_key='edge_id')

        if demands_file_path:
            self.load_demands(demands_file_path)

        self.recalc_derived_data()


    def get_cost_function(self):
        N: int = self.graph.number_of_edges()
        free_flows = np.ndarray((N,), dtype=float)
        koeffs = np.ndarray((N,), dtype=float)
        capacity_inv = np.ndarray((N,), dtype=float)
        power = np.ndarray((N,), dtype=float)
        for i, e in enumerate(self.graph.edges.data()):
            free_flows[i] = e[2]['frf']
            koeffs[i] = e[2]['k']
            capacity_inv[i] = 1. / e[2]['cap']
            power[i] = e[2]['pow']

        def cost(x: np.ndarray) -> np.ndarray:
            return self.Q.T @ (free_flows * (1.0 + koeffs * np.power(((self.Q @ x) * capacity_inv), power)))

        return cost
