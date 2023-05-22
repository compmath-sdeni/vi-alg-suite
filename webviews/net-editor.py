import dash
import dash_cytoscape as cyto
from dash import html, dcc
from dash.dependencies import Input, Output, State
import networkx as nx

from methods.algorithm_params import AlgorithmParams
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
from problems.testcases.blood_delivery import blood_delivery_hardcoded_test_one, blood_delivery_test_one, \
    blood_delivery_test_two, blood_delivery_test_three

# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/cytoscape/events

# Create the graph using NetworkX
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4, 5])
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

# Initialize node positions
pos = nx.spring_layout(G)
print(pos)
pos = [(100, 100), (200, 100), (100, 200), (200, 200), (100, 300), (200, 300)]

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

problem = blood_delivery_test_two.prepareProblem(algorithm_params=params, show_network=False)
G, pos, labels = problem.net.to_nx_graph()

# Create the Dash application
app = dash.Dash(__name__)
#                , external_stylesheets=["netedit.css"]

# Define the Dash layout
app.layout = html.Div(
    html.Div(
    children=[
        # div containing the graph, should take half of the screen
        html.Div(style={"width": "45%", "display": "inline-block"}, children = [
        cyto.Cytoscape(
            id="graph_presenter",
            layout={"name": "preset"},
            style={"width": "800px", "height": "600px", "border": "2px solid green"},
            elements=[
                         {"data": {"id": str(node), "label": "Nod " + str(i)},
                          "position": {"x": pos[node][0], "y": pos[node][1]}} for i, node in enumerate(G.nodes())
                     ] + [
                         {"data": {"source": str(edge[0]), "target": str(edge[1])}} for edge in G.edges()
                     ],
            stylesheet=[
                {
                    "selector": 'node',
                    "style": {
                        'background-color': '#038',
                        'label': 'data(id)'
                    }
                },
                {
                    "selector": 'edge',
                    "style": {
                        'width': 3,
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }
                }
            ]
        )]),

        html.Div(style={"width": "45%", "height":"100%", "display": "inline-block", "border": "1px solid gray", "padding":"8px"}, children = [
            html.Div([
                html.H2("Edge Manipulation"),
                html.Div([
                    html.Label("New Edge"),
                    dcc.Input(id='source-node-input', type='text', placeholder="Source node"),
                    dcc.Input(id='target-node-input', type='text', placeholder="Target node"),
                    html.Button("Add Edge", id='add-edge-button'),
                ]),
                html.Div([
                    html.Label("Remove Edge"),
                    dcc.Input(id='edge-input', type='text', placeholder="Source-Target"),
                    html.Button("Remove Edge", id='remove-edge-button'),
                ]),
                html.Div([
                    html.Label("Set Edge Weight"),
                    dcc.Input(id='edge-weight-input', type='text', placeholder="Source-Target-Weight"),
                    html.Button("Set Weight", id='set-weight-button'),
                ]),
            ]),

            html.Div(id='console-output'),
            html.Div(),
            html.Button("Save", id="save-button", disabled=True),
        ]),

        ])
    )


@app.callback(Output('console-output', 'children'),
              [Input('graph_presenter', 'tapEdgeData'), Input('graph_presenter', 'tapNodeData')]
              )
def onEdgeClick(edgeData, nodeData):
    print(dash.ctx.triggered)
    if edgeData:
        return "You recently clicked/tapped the edge between " + \
            edgeData['source'].upper() + " and " + edgeData['target'].upper()
    elif nodeData:
        G.add_node()
        return "You recently clicked/tapped the node " + nodeData['id'].upper()


@app.callback(
    Output("save-button", "disabled"),
    Input("graph_presenter", "elements"),
)
def enable_save_button(elements):
    if elements:
        return False
    return True


# @app.callback(
#     Output('graph_presenter', 'elements'),
#     Input("add-edge-button", "n_clicks"),
# )
# def test_callback(elements, n_clicks):
#     return elements


# app.callback(
#     Output('graph_presenter', 'elements'),
#     [
#         Input('add-edge-button', 'n_clicks'),
#         Input('remove-edge-button', 'n_clicks'),
#         Input('set-weight-button', 'n_clicks')
#     ],
#     [
#         State('edge-input', 'value'),
#         State('edge-weight-input', 'value'),
#         State('source-node-input', 'value'),
#         State('target-node-input', 'value'),
#         State('graph_presenter', 'elements')
#     ]
# )
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
