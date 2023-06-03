import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import networkx as nx

from methods.algorithm_params import AlgorithmParams
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
from problems.testcases.blood_delivery import blood_delivery_hardcoded_test_one, blood_delivery_test_one, \
    blood_delivery_test_two, blood_delivery_test_three

# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/cytoscape/events

active_edge = None
active_node = None

def prepare_problem():
    params = AlgorithmParams(
        eps=1e-5,
        min_iters=10,
        max_iters=500,
        lam=0.01,
        lam_KL=0.005,
        start_adaptive_lam=0.5,
        start_adaptive_lam1=0.5,
        adaptive_tau=0.75,
        adaptive_tau_small=0.45,
        save_history=True,
        excel_history=True
    )

    problem = blood_delivery_test_three.prepareProblem(algorithm_params=params, show_network=False, print_data=False)
    G, pos, labels = problem.net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0)

    return G, pos, labels

if __name__ == "__main__":
    G, pos, labels = prepare_problem()

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")

# Create the Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
#                , external_stylesheets=["netedit.css"]

# Define the Dash layout
app.layout = html.Div(
    className="dbc container-fluid vh-100",
    children=[
        dbc.Row(
            className="vh-100",
            children=[
            # div containing the graph, should take half of the screen
            dbc.Col(style={"border": "1px solid green"}, children = [
            cyto.Cytoscape(
                id="graph_presenter",
                layout={"name": "preset"},
                style={"width": "98%", "height": "98%"},
                elements=[
                             {"data": {"id": str(node), "label": "Nod " + str(i)},
                              "position": {"x": pos[node][0], "y": pos[node][1]}} for i, node in enumerate(G.nodes())
                         ] + [
                             {"data": {"source": str(edge[0]), "target": str(edge[1]), "edge_label": str(idx)}} for idx, edge in enumerate(G.edges())
                         ],
                stylesheet=[
                    {
                        "selector": 'node',
                        "style": {
                            'background-color': '#BBBBFF',
                            'text-halign':'center',
                            'text-valign':'center',
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
            )]),

            dbc.Col(style={"border": "1px solid gray"}, children = [
                html.H2("Edge Manipulation", className="bg-primary text-white p-2 mb-2 text-center"),
                html.Form(className="form",
                    children=[
                        html.Div(className="row align-items-left  mt-2",
                            children=[
                                html.Div(className="col-sm-3", children=[
                                    dcc.Input(id='source-node-input', type='text', placeholder="Source node", className="form-control"),
                                ]),
                                html.Div(className="col-sm-3", children=[
                                    dcc.Input(id='target-node-input', type='text', placeholder="Target node", className="form-control"),
                                ]),
                                html.Div(className="col-sm-3", children=[
                                    dbc.Button("Add Edge", id='add-edge-button', className="btn btn-primary"),
                                ]),
                                html.Div(className="col-sm-3", children=[
                                    html.Button("Remove Edge", id='remove-edge-button', className="btn btn-danger"),
                                ]),
                            ]),
                        html.Div(className="row align-items-left mt-2",
                                 children=[
                                     html.Div(className="col-sm-3", children=[
                                         dcc.Input(id='edge-oper-cost-input', type='text', placeholder="unit operational cost",
                                                   className="form-control"),
                                     ]),
                                     html.Div(className="col-sm-3", children=[
                                         html.Button("Set", id='set-oper-cost-button', className="btn btn-info"),
                                     ]),
                                     html.Div(className="col-sm-3", children=[
                                         dcc.Input(id='edge-discard-cost-input', type='text',
                                                   placeholder="unit discard cost",
                                                   className="form-control"),
                                     ]),
                                     html.Div(className="col-sm-3", children=[
                                         html.Button("Set", id='set-discard-cost-button', className="btn btn-info"),
                                     ]),
                                 ]),
                    ]),

                html.Div(id='console-output'),
            ])
        ])
    ])

@app.callback(Output('console-output', 'children'),
              [Input('graph_presenter', 'tapEdgeData'), Input('graph_presenter', 'tapNodeData')]
              )
def onGraphElementClick(edgeData, nodeData):
    global active_edge, active_node

    context = dash.ctx.triggered
    print(f"Context: {context}")
    print(f"Edge data: {edgeData}")
    print(f"Node data: {nodeData}")
    if context[0]['prop_id'] == 'graph_presenter.tapEdgeData' and edgeData:
        active_edge = edgeData
        active_node = None

        return "onGraphElementClick: clicked/tapped the edge between " + \
            edgeData['source'].upper() + " and " + edgeData['target'].upper()
    elif context[0]['prop_id'] == 'graph_presenter.tapNodeData' and nodeData:
        active_node = nodeData
        active_edge = None
        return "onGraphElementClick: clicked/tapped the node " + nodeData['id'].upper()



# @app.callback(
#     Output('graph_presenter', 'elements'),
#     Input("add-edge-button", "n_clicks"),
# )
# def test_callback(elements, n_clicks):
#     return elements


@app.callback(
    Output('graph_presenter', 'elements'),
    [
        Input('remove-edge-button', 'n_clicks')
    ],
    [
        State('graph_presenter', 'elements')
    ]
)
def remove_edge_callback(n_clicks, elements):
    global active_edge

    if n_clicks is None:
        raise PreventUpdate

    if active_edge is None:
        raise PreventUpdate

    print(f"remove_edge_callback: removing edge {active_edge}")

    source = active_edge['source']
    target = active_edge['target']

    print(f"Removing edge between {source} and {target}")

    # for elem in elements:
    #     print(f"elem: {elem}")

    # remove edge from networkx graph
    G.remove_edge(int(source), int(target))

    elements = [elem for elem in elements if ((not 'source' in elem['data']) or (not (elem['data']['source'] == source and elem['data']['target'] == target)))]

    return elements


#
#
# def graph_modify(
#         add_edge_clicks, remove_edge_clicks, set_weight_clicks,
#         edge_input, edge_weight, source_node, target_node,
#         elements):
#     # print(f"add_edge_clicks: {add_edge_clicks}, remove_edge_clicks: {remove_edge_clicks}, "
#     #       f"set_weight_clicks: {set_weight_clicks}. edge_input: {edge_input}, edge_weight: {edge_weight}, "
#     #       f"source_node: {source_node}, target_node: {target_node}")
#
#     # if add_edge_clicks is not None:
#     #     new_edge = {'data': {'source': source_node, 'target': target_node}}
#     #     elements.append(new_edge)
#     # elif remove_edge_clicks is not None:
#     #     elements = [e for e in elements if e['data']['source'] != source_node or e['data']['target'] != target_node]
#     # elif set_weight_clicks is not None:
#     #     source_node, target_node, weight = edge_weight.split("-")
#     #     for e in elements:
#     #         if e['data']['source'] == source_node and e['data']['target'] == target_node:
#     #             e['data']['weight'] = weight
#
#     return elements
#

if __name__ == "__main__":
    app.run_server(debug=True)
