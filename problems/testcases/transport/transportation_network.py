import math
import os
import sys
import time
from collections.abc import Iterable
from enum import Enum, unique
from operator import itemgetter

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
                 demand: List[Tuple[int, int, float]] = None,
                 max_od_paths_count: int = 3,
                 max_path_edges: int = 10,
                 nodes_coords: List[Tuple[int, float, float]] = None) -> None:

        self.paths_count = 0
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.paths: List[List[np.ndarray]] = []
        self.Q: np.ndarray = None
        self.total_demand: float = 0
        self.nodes_coords = nodes_coords
        self.keyed_edges = {}
        self.all_possible_paths_cache: List[List[np.ndarray]] = None

        self.max_od_paths_count = max_od_paths_count
        self.max_path_edges = max_path_edges

        if network_graph_file:
            self.load_network_graph(network_graph_file)

        if edges_list:
            sorted_edges = sorted(edges_list, key=itemgetter(0, 1))

            k: int = 0  # edge key = zero based edge number for algs
            for e in sorted_edges:
                self.graph.add_edge(e[0], e[1], key=k, **e[2])
                self.keyed_edges[k] = (e[0], e[1], k)
                k += 1

        if demand:
            self.demand: List[Tuple[int, int, float]] = demand
        else:
            self.demand: List[Tuple[int, int, float]] = []

        if edges_list and demand:
            self.calc_paths()

    def show(self, *, limit: int = 7, print_edges_data: bool = False):
        print(self.graph)
        print("Edges:")
        for k in self.keyed_edges.keys():
            if k > 0:
                print('; ', end='')
            print(f"{k}: {self.keyed_edges[k][:2]}", end='')
        print()

        print(f"Demands (total {len(self.demand)}): ")
        for d in self.demand[:limit]:
            print(f'{d[0]} -> {d[1]}: {d[2]}')

        # edg = list(self.graph.edges(data=True))

        pi = 0
        for idx, paths_for_pair in enumerate(self.paths):
            if idx > 0:
                print()
            print(f'Paths for {self.demand[idx][:2]}', sep='', end='')
            for l, path_edges in enumerate(paths_for_pair):
                if l > limit:
                    break

                print(f"\n{pi}: ", sep='', end='')
                pi += 1
                for k, edge_key in enumerate(path_edges):
                    if k == 0:
                        print(self.keyed_edges[edge_key][0], sep='', end='')
                    print('->', sep='', end='')
                    print(self.keyed_edges[edge_key][1], sep='', end='')

        print()

    def draw(self):
        if self.nodes_coords:
            pos = self.nodes_coords
        else:
            pos = nx.planar_layout(self.graph)

        edge_labels = {}
        for e in self.graph.edges:
            data = self.graph.edges[e[0], e[1], e[2]]
            pw = data['pow']
            lbl = ''
            if pw != 1:
                lbl = f"{data['frf']}(1+{data['k']}(y/{data['cap']})^{data['pow']})"
            else:
                lbl = f"{data['frf']}(1+{data['k']}(y/{data['cap']})^{data['pow']})"
                # cutoff = 0.0000001
                # frf = data['frf']
                # ky = frf * data['k'] / data['cap']
                # if math.fabs(ky - math.floor(ky)) < cutoff:
                #     ky = math.floor(ky)
                # if math.fabs(frf - math.floor(frf)) < cutoff:
                #     frf = math.floor(frf)
                #
                # if frf >= cutoff:
                #     lbl = f"{frf}"
                #
                # if ky >= cutoff:
                #     if lbl:
                #         lbl += '+'
                #     if ky != 1:
                #         lbl = f"{ky}y"
                #     else:
                #         lbl = f"y"

            edge_labels[(e[0], e[1])] = lbl

        nx.draw(self.graph, pos, labels={node: node for node in self.graph.nodes()})
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

    # def recalc_derived_data(self, *, saved_paths_file: str = None, max_od_paths_count: int = 3,
    #                         max_path_edges: int = 10, cached_paths_file: str = None):
    #     self.calc_paths(saved_paths_file=saved_paths_file,
    #                     max_od_paths_count=max_od_paths_count, max_path_edges=max_path_edges,
    #                     cached_paths_file=cached_paths_file)
    #
    #     self.Q = self._calc_edges_to_paths_incidence_()

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

    def select_best_paths_for_pair(self, *, od_paths: List, count: int, traffic_on_edges: np.ndarray = None):
        # TODO: use traffic!
        def get_path_cost(path):
            cost = 0
            for edg_key in path:
                e = self.graph.edges[self.keyed_edges[edg_key]]

                if traffic_on_edges is None:
                    small_cost = self.get_edge_cost_by_flow(e, 1.0)
                    big_cost = self.get_edge_cost_by_flow(e, 100000.)

                    cost += small_cost * 10 + math.log(big_cost)
                else:
                    self.get_edge_cost_by_flow(e, traffic_on_edges[edg_key])

            return cost

        sorted_paths = sorted(od_paths, key=get_path_cost)

        return sorted_paths[:count]

    def select_best_paths(self, possible_paths, max_od_paths_count, traffic_on_edges):
        start = time.process_time()
        paths_count = 0
        paths = []
        total_cnt = 0
        for idx, paths_for_pair_edge_ids in enumerate(possible_paths):
            total_cnt += len(paths_for_pair_edge_ids)
            if len(paths_for_pair_edge_ids) > max_od_paths_count:
                suitable_paths = self.select_best_paths_for_pair(od_paths=paths_for_pair_edge_ids,
                                                        count=max_od_paths_count, traffic_on_edges=traffic_on_edges)
            else:
                suitable_paths = paths_for_pair_edge_ids

            paths_count += len(suitable_paths)
            paths.append(suitable_paths)

        end = time.process_time()
        print(f"select_best_paths: selected {paths_count} 'best' paths of {total_cnt} in {end - start} sec.")
        return (paths, paths_count)

    def calc_paths(self, *, saved_paths_file: str = None, max_od_paths_count: int = None, max_path_edges: int = None,
                   cached_paths_file: str = None, traffic_on_edges: np.ndarray = None):
        self.paths = []
        self.paths_count = 0
        self.Q = None

        if max_od_paths_count:
            self.max_od_paths_count = max_od_paths_count
        else:
            max_od_paths_count = self.max_od_paths_count

        if max_path_edges:
            self.max_path_edges = max_path_edges
        else:
            max_path_edges = self.max_path_edges

        if saved_paths_file and os.path.exists(saved_paths_file):
            start = time.process_time()
            self.paths = np.load(saved_paths_file, allow_pickle=True)
            for p in self.paths:
                self.paths_count += len(p)
            end = time.process_time()
            print(f"calc_paths: loaded {self.paths_count} paths in {end - start} sec.")
        else:
            if self.all_possible_paths_cache is not None:
                paths = self.all_possible_paths_cache
            elif cached_paths_file and os.path.exists(cached_paths_file):
                # Load all possible paths of max edge count from "cache"
                start = time.process_time()
                paths = np.load(cached_paths_file, allow_pickle=True)
                end = time.process_time()
                print(f"calc_paths: loaded {len(paths)} possible paths from cache in {end - start} sec.")

                self.all_possible_paths_cache = paths
            else:
                # calculate all simple paths of max. edges count and save to cache
                start = time.process_time()
                paths = []
                cnt = 0
                for d in self.demand:
                    paths_for_pair_edge_ids: List = []

                    # startp = time.process_time()
                    possible_paths = list(nx.all_simple_edge_paths(self.graph, d[0], d[1], cutoff=max_path_edges))
                    # endp = time.process_time()
                    # print(f"Found {len(possible_paths)} simple paths of {max_path_edges} edges max from {d[0]} to {d[1]}")

                    for p in possible_paths:
                        edge_keys = np.ndarray((len(p),), dtype=int)
                        i = 0
                        for e in p:
                            edge_keys[i] = e[2]
                            i += 1
                        paths_for_pair_edge_ids.append(edge_keys)
                        cnt += 1

                    paths.append(paths_for_pair_edge_ids)

                end = time.process_time()
                print(f"calc_paths: calculated {cnt} possible paths of {max_path_edges} edges max in {end - start} sec.")

                self.all_possible_paths_cache = paths

                if cached_paths_file:
                    np.save(cached_paths_file, np.array(paths, dtype=object), allow_pickle=True)

            # select supposedly best paths
            self.paths, self.paths_count = self.select_best_paths(paths, max_od_paths_count, traffic_on_edges)
            if saved_paths_file:
                np.save(saved_paths_file, np.array(self.paths, dtype=object), allow_pickle=True)
                print(f"calc_paths: saved {self.paths_count} 'best' paths.")

        self.Q = self._calc_edges_to_paths_incidence_()

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

    def load_positions(self, file_path: str):
        pos = {}
        f = open(file_path, 'r')
        file_contents = f.read()
        rows = file_contents.splitlines()
        for row in rows:
            parts = row.strip().split(' ')
            pos[int(parts[0])] = (int(parts[1]), int(parts[2]))

        return pos

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
                if k != orig and destinations[k] > 0:
                    self.demand.append((orig, k, destinations[k]))
                    self.total_demand += destinations[k]

            matrix[orig] = destinations
        zones = max(matrix.keys())
        mat: np.ndarray = np.zeros((zones, zones))
        for i in range(zones):
            for j in range(zones):
                # We map values to a index i-1, as Numpy is base 0
                mat[i, j] = matrix.get(i + 1, {}).get(j + 1, 0)

    def load_network_graph(self, edges_list_file_path: str, demands_file_path: str, *, saved_paths_file: str = None,
                           pos_file: str = None, columns_separator: str = '\t',
                           max_od_paths_count: int = 3, max_path_edges: int = 10, cached_paths_file: str = None):
        net: pd.DataFrame = pd.read_csv(edges_list_file_path, skiprows=8, sep=columns_separator, skipinitialspace=False)

        trimmed = [s.strip().lower() for s in net.columns]
        net.columns = trimmed

        # And drop the silly first and last columns
        net.drop(['~', ';'], axis=1, inplace=True)

        # net['link_type'] = net['link_type'].map(lambda x: x.rstrip(' ;'))

        net.rename(columns={'free_flow_time': str(EdgeParams.FRF), 'capacity': str(EdgeParams.CAP),
                            'b': str(EdgeParams.K), 'power': str(EdgeParams.POW), 'length': str(EdgeParams.LEN)},
                   inplace=True)

        net.sort_values(by=['init_node', 'term_node'], inplace=True)

        k: int = 0  # edge key = zero based edge number for algs
        for e in net.itertuples(False):
            ed = e._asdict()
            src = ed.pop('init_node')
            dst = ed.pop('term_node')
            self.graph.add_edge(src, dst, key=k, **ed)
            self.keyed_edges[k] = (e[0], e[1], k)
            k += 1

        # net['edge_id'] = np.arange(len(net))
        #
        # self.graph = nx.from_pandas_edgelist(net, 'init_node', 'term_node',
        #                         [str(EdgeParams.FRF), str(EdgeParams.CAP), str(EdgeParams.K),
        #                          str(EdgeParams.POW), str(EdgeParams.LEN)],
        #                         create_using=nx.MultiDiGraph, edge_key='edge_id')

        if pos_file is not None:
            self.nodes_coords = self.load_positions(pos_file)

        k: int = 0  # edge key = zero based edge number for algs
        for e in self.graph.edges:
            self.keyed_edges[e[2]] = (e[0], e[1], e[2])
            k += 1

        if demands_file_path:
            self.load_demands(demands_file_path)

        self.calc_paths(saved_paths_file=saved_paths_file, max_od_paths_count=max_od_paths_count,
                                 max_path_edges=max_path_edges, cached_paths_file=cached_paths_file)

    def get_edge_cost_by_flow(self, e, flow):
        free_flow = e[str(EdgeParams.FRF)]
        koeff = e[str(EdgeParams.K)]
        capacity_inv = 1. / e[str(EdgeParams.CAP)]
        power = e[str(EdgeParams.POW)]

        return free_flow * (1.0 + koeff * np.power(flow * capacity_inv, power))

    def get_cost_function(self):
        N: int = self.graph.number_of_edges()
        free_flows = np.ndarray((N,), dtype=float)
        koeffs = np.ndarray((N,), dtype=float)
        capacity_inv = np.ndarray((N,), dtype=float)
        power = np.ndarray((N,), dtype=float)

        has_non_one_power = False

        # for i, e in enumerate(self.graph.edges.data()):
        for k, edg_key in self.keyed_edges.items():
            e = self.graph.edges[edg_key]
            free_flows[k] = e[str(EdgeParams.FRF)]
            koeffs[k] = e[str(EdgeParams.K)]
            capacity_inv[k] = 1. / e[str(EdgeParams.CAP)]
            power[k] = e[str(EdgeParams.POW)]

        if np.all(np.isclose(power, np.ones_like(power))):
            def cost(x: np.ndarray) -> np.ndarray:
                return self.Q.T @ (free_flows * (1.0 + koeffs * (self.Q @ x) * capacity_inv))
        else:
            def cost(x: np.ndarray) -> np.ndarray:
                return self.Q.T @ (free_flows * (1.0 + koeffs * np.power(((self.Q @ x) * capacity_inv), power)))

        return cost
