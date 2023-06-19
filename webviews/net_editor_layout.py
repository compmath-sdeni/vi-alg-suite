from typing import List, Dict

import dash_cytoscape as cyto
from dash import html, dcc
import networkx as nx

from problems.blood_supply_net_problem import BloodSupplyNetworkProblem, BloodSupplyNetwork

import logging

logger = logging.getLogger('vi-algo-test-suite')


def get_saved_problems_list():
    return [
        {"label": "Blood delivery test one", "value": "blood_delivery_test_one"},
        {"label": "Blood delivery test two", "value": "blood_delivery_test_two"},
        {"label": "Blood delivery test three", "value": "blood_delivery_test_three"},
    ]


def get_cytoscape_graph_elements(net: BloodSupplyNetwork, *, G: nx.Graph = None, pos: dict = None, labels: dict = None):
    if G is None:
        G, pos, labels = net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0)

    try:
        return [
            {"data": {"id": str(node), "label": "Nod " + str(i)},
             "position": {"x": pos[node][0], "y": pos[node][1]}} for i, node in
            enumerate(G.nodes())
        ] + [
            {
                "data": {
                    "source": str(edge[0]), "target": str(edge[1]), "edge_index": str(idx), "edge_label": str(idx),
                    "operational_cost": net.c_string[idx], "waste_discard_cost": net.z_string[idx],
                    "risk_cost": (net.r_string[idx] if len(net.r_string) > idx else ''),
                    "alpha": net.edge_loss[idx],
                }
            } for idx, edge in enumerate(G.edges())
        ]
    except Exception as e:
        logger.error(f"Error in get_cytoscape_graph_elements: {e}")
        for i, node in enumerate(G.nodes()):
            print(i, node, pos[node][0], pos[node][1])

        for idx, edge in enumerate(G.edges()):
            print(idx, edge[0], edge[1], net.c_string[idx], net.z_string[idx],
                  (net.r_string[idx] if len(net.r_string) > idx else ''), net.edge_loss[idx])
        return []


def update_net_by_cytoscape_elements(graph_elements: List[Dict], net: BloodSupplyNetwork):
    for elem in graph_elements:
        data = elem['data']
        if 'source' in data and 'target' in data:  # edge element
            idx = int(data['edge_index'])
            net.c_string[idx] = data['operational_cost']
            net.z_string[idx] = data['waste_discard_cost']
            if ('risk_cost' in data) and data['risk_cost']:
                net.r_string[idx] = data['risk_cost']

            net.edge_loss[idx] = float(data['alpha'])
        elif 'id' in data:  # node element
            if 'position' in elem:
                net.pos[int(data['id'])] = (elem['position']['x'], elem['position']['y'])


def build_graph_view_layout(net: BloodSupplyNetwork, G, pos, labels):
    return cyto.Cytoscape(
        id="graph_presenter",
        layout={"name": "preset"},
        style={"width": "98%", "height": "98%"},
        elements=get_cytoscape_graph_elements(net, G=G, pos=pos, labels=labels),
        stylesheet=[
            {
                "selector": 'node',
                "style": {
                    'background-color': '#BBBBFF',
                    'text-halign': 'center',
                    'text-valign': 'center',
                    'label': 'data(id)'
                }
            },
            {
                "selector": 'edge',
                "style": {
                    'source-label': 'data(edge_label)',
                    'source-text-offset': '20px',
                    'width': 1,
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ]
    )

def get_layout(problem: BloodSupplyNetworkProblem, session_id: str):
    logger.info(f"get_layout called - creating application layout. Session: {session_id}")

    G, pos, labels = problem.net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0, update_positions=True)

    return html.Div(
        className="dbc container-fluid vh-100",
        children=[
            dcc.Store(data=session_id, id='session-id', storage_type='session'),
            dcc.Input(id='session-id-input', type='hidden', value=session_id),
            dcc.Input(id='selected-node-id', type='hidden', value=''),
            dcc.Input(id='selected-edge-index', type='hidden', value=''),
            dcc.Input(id='temp-data', type='hidden', value=''),
            dcc.Input(id='temp-data-target', type='hidden', value=''),
            html.Div(
                className="row vh-100",
                children=[
                    # div containing the graph, should take half of the screen
                    html.Div(className="col-sm-5", id="graph-container", style={"border": "1px solid green"},
                             children=build_graph_view_layout(problem.net, G, pos, labels)),
                    html.Div(className="col-sm-7", style={"border": "1px solid gray"}, children=[
                        html.H4("User and session", className="bg-info text-white p-2 mb-2 text-center"),
                        html.Div(className="form", children=[
                            html.Div(id="login-form-block", children=[
                                html.Div(className="row align-items-left mt-2", children=[
                                    html.Div(className="col-sm-3", children=[
                                        dcc.Input(id='email-input', type='text',
                                                  placeholder="Email", className="form-control"),
                                    ]),
                                    html.Div(className="col-sm-3", children=[
                                        dcc.Input(id='password-input', type='password',
                                                  placeholder="password", className="form-control"),
                                    ]),
                                    html.Div(className="col-sm-3", children=[
                                        html.Button("Init/continue session", id='login-button',
                                                    className="btn btn-primary"),
                                    ]),
                                ])
                            ]),
                            html.Div(id="user-session-block", style={"display": "none"}, children=[
                                html.Div(className="row align-items-left mt-2 ml-2", children=[
                                    html.Div(className="col-sm-2 mt-1", children=[
                                        html.Span("Hello, "),
                                        html.Span(id='user-email-show',
                                                  style={"fontWeight": "bold", "fontSize": "larger"}),
                                    ]),
                                    html.Div(className="col-sm-2", children=[
                                        html.Button("Log out", id='logout-button',
                                                    className="btn btn-warning"),
                                    ]),
                                    html.Div(className="col-sm-6", children=[
                                        dcc.Dropdown(id='user-saved-problems', className="form-select",
                                                     options=[{'value': problem["value"], 'label': problem["value"]} for
                                                              problem in get_saved_problems_list()], value=None,
                                                     placeholder="Select problem")
                                    ]),
                                    html.Div(className="col-sm-2", children=[
                                        html.Button("Load problem", id='load-problem-button',
                                                    className="btn btn-primary")
                                    ]),
                                ])
                            ]),
                            html.Div(id="login-error-block", className="alert alert-danger",
                                     style={"display": "none", "textAlign": "center"})
                        ]),

                        html.H4("Problem editor", className="bg-primary text-white p-2 mb-2 mt-2 text-center"),
                        html.Div(className="form",
                                 children=[
                                     html.Div([html.H4("Edge data", className="my-0 py-0")],
                                              className="form row align-items-left mt-2 g-1"),
                                     html.Div(className="form row align-items-left mt-2 g-1",
                                              children=[
                                                  html.Div(className="col-sm-1", children=[
                                                      html.Label("Source", htmlFor="source-node-input",
                                                                 className="form-label"),
                                                      dcc.Input(id='source-node-input', type='text',
                                                                placeholder="1", className="form-control"),
                                                  ]),
                                                  html.Div(className="col-sm-1", children=[
                                                      html.Label("Dest.", htmlFor="target-node-input",
                                                                 className="form-label"),
                                                      dcc.Input(id='target-node-input', type='text',
                                                                placeholder="2", className="form-control"),
                                                  ]),
                                                  html.Div(className="col-sm-3", children=[
                                                      html.Label("Operational cost: c(f); c'(f)",
                                                                 htmlFor="oper-cost-input",
                                                                 className="form-label"),
                                                      html.Div(className="row g-1", children=[
                                                          html.Div(className="col-sm-8", children=[
                                                              dcc.Input(id='oper-cost-input', type='text',
                                                                        placeholder="e.g. f^2+3*f+1",
                                                                        className="form-control"),
                                                          ]),
                                                          html.Div(className="col-sm-4", children=[
                                                              dcc.Input(id='oper-cost-deriv-input', type='text',
                                                                        placeholder="2*f+3",
                                                                        className="form-control"),
                                                          ])
                                                      ]),
                                                  ]),
                                                  html.Div(className="col-sm-3", children=[
                                                      html.Label("Waste cost: z(f); z'(f)",
                                                                 htmlFor="waste-discard-cost-input",
                                                                 className="form-label"),
                                                      html.Div(className="row g-1", children=[
                                                          html.Div(className="col-sm-7", children=[
                                                              dcc.Input(id='waste-discard-cost-input', type='text',
                                                                        placeholder="e.g. 0.75*f",
                                                                        className="form-control"),
                                                          ]),
                                                          html.Div(className="col-sm-5", children=[
                                                              dcc.Input(id='waste-discard-cost-deriv-input',
                                                                        type='text',
                                                                        placeholder="0.75",
                                                                        className="form-control"),

                                                          ])
                                                      ]),

                                                  ]),
                                                  html.Div(className="col-sm-3", children=[
                                                      html.Label("Risk cost: r(f); r'(f)", htmlFor="risk-cost-input",
                                                                 className="form-label"),
                                                      html.Div(className="row g-1", children=[
                                                          html.Div(className="col-sm-6", children=[
                                                              dcc.Input(id='risk-cost-input', type='text',
                                                                        placeholder="e.g. 1.5*f",
                                                                        className="form-control"),
                                                          ]),
                                                          html.Div(className="col-sm-6", children=[
                                                              dcc.Input(id='risk-cost-deriv-input', type='text',
                                                                        placeholder="1.5",
                                                                        className="form-control"),
                                                          ]),
                                                      ]),
                                                  ]),
                                                  html.Div(className="col-sm-1", children=[
                                                      html.Label("1-loss", htmlFor="edge-loss-input",
                                                                 className="form-label"),
                                                      dcc.Input(id='edge-loss-input', type='text',
                                                                placeholder="e.g. 0.9",
                                                                className="form-control"),
                                                  ]),
                                              ]),
                                     html.Div(className="form row align-items-left mt-2 g-1",
                                              children=[
                                                  html.Div(className="col-auto", children=[
                                                      html.Button("Add Edge", id='add-edge-button',
                                                                  className="btn btn-primary"),
                                                  ]),

                                                  html.Div(className="col-auto", children=[
                                                      html.Button("Set edge parameters", id='set-edge-params-button',
                                                                  className="btn btn-info"),
                                                  ]),

                                                  html.Div(className="col-auto", children=[
                                                      html.Button("Remove Edge", id='remove-edge-button',
                                                                  className="btn btn-danger"),
                                                  ]),
                                              ]),
                                     html.Hr(),
                                     html.Div([html.H4("Node data", className="my-0 py-0")],
                                              className="form row align-items-left mt-2 g-1"),
                                     html.Hr(),
                                     html.Div([html.H4("Save problem", className="my-0 py-0")],
                                              className="form row align-items-left mt-2 g-1"),
                                     html.Div("Name will be name of folder with the problem data",
                                              className="form row align-items-left mt-2 g-1"),
                                     html.Div(className="form row align-items-left mt-2 g-1",
                                              children=[
                                                  html.Div(className="col-sm-4", children=[
                                                      dcc.Input(id='save-problem-name-input', type='text',
                                                                placeholder="problem name", className="form-control"),
                                                  ]),
                                                  html.Div(className="col-auto", children=[
                                                      html.Button("Save", id='save-problem-button',
                                                                  className="btn btn-success"),
                                                  ]),
                                                  html.Div(className="col-auto", children=[
                                                      html.Div(id="save-error-block", className="alert alert-danger",
                                                               style={"display": "none", "textAlign": "center"})
                                                  ])
                                              ])
                                 ]),
                        html.H4("Console output", className="bg-info p-1 mb-1 mt-4 text-center"),
                        html.Div(id='console-output'),
                    ])
                ])
        ])
