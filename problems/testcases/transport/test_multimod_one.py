import numpy as np

from methods.algorithm_params import AlgorithmParams
from methods.korpele_adaptive import KorpelevichExtragradAdapt
from methods.tseng_adaptive import TsengAdaptive
from methods.extrapol_from_past_adaptive import ExtrapolationFromPastAdapt
from problems.multi_traffic_eq_problem import MultiModalTrafficEquilibriumProblem
from problems.testcases.transport.multi_transport_net import MultiModalTransportationNetwork

def prepareProblem(*, algorithm_params: AlgorithmParams = AlgorithmParams()):

    # Example multi-modal edges (simplified)
    edges_list = [
        (1, 2, {'mode': 'bike', 'free_flow_time': 10, 'capacity': 100, 'length': 2}),
        (2, 3, {'mode': 'bus', 'free_flow_time': 5, 'capacity': 50, 'length': 3}),
        (1, 3, {'mode': 'car', 'free_flow_time': 3, 'capacity': 30, 'length': 1}),
    ]

    # Modes parameters based on your article (simplified)
    modes_info = {
        'bike': {'free_flow_time': 10, 'alpha': 0.15, 'beta': 4, 'capacity': 100, 'crowd_discomfort': 0.6, 'fare': 0.1,
                 'lambda_t': 0.5, 'lambda_u': 0.3, 'lambda_m': 0.2},
        'bus': {'free_flow_time': 5, 'alpha': 0.15, 'beta': 4, 'capacity': 50, 'crowd_discomfort': 0.5, 'fare': 0.15,
                'lambda_t': 0.5, 'lambda_u': 0.3, 'lambda_m': 0.2},
        'car': {'free_flow_time': 3, 'alpha': 0.15, 'beta': 4, 'capacity': 30, 'crowd_discomfort': 0.1, 'fare': 0.2,
                'lambda_t': 0.5, 'lambda_u': 0.3, 'lambda_m': 0.2}
    }

    # Uncertain demand intervals
    demands_interval = np.array([[3800, 4200], [9500, 10500]])

    n = 3
    algorithm_params.x0 = np.array([0. for i in range(n)])

    # Create the multimodal network
    multimodal_network = MultiModalTransportationNetwork(edges_list=edges_list, demand=[(1,3,4000)], modes_info=modes_info)

    # Setup the variational inequality problem
    problem = MultiModalTrafficEquilibriumProblem(multimodal_network, demands_interval, x0=algorithm_params.x0)

    return problem
