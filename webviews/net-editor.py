import hashlib
import uuid

import flask
from flask_login import LoginManager
from flask_caching import Cache
import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc, Patch
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import networkx as nx

from methods.algorithm_params import AlgorithmParams
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
from problems.testcases.blood_delivery import blood_delivery_hardcoded_test_one, blood_delivery_test_one, \
    blood_delivery_test_two, blood_delivery_test_three

import os
import dotenv

# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/cytoscape/events

# https://community.plotly.com/t/how-to-update-cytoscape-elements-list-of-dics-using-patch/75631

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


dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")

# Create the Dash application based on Flask server
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,
                title='VI algorithms test suite',
                update_title='Loading...',
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css]
                )
#                , external_stylesheets=["netedit.css"]

flask_cache_config = {
    'CACHE_TYPE': 'SimpleCache'
    # try 'FileSystemCache' if you don't want to setup redis
    # 'cache_type': 'redis',
    # 'cache_redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}

cache = Cache(config=flask_cache_config)

# Configure flask login with secret key from environment variable
server.config.update(SECRET_KEY=os.getenv('WEB_APP_SECRET_KEY'))

login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

# Define the Dash layout

def get_saved_problems_list():
    return [
        {"label": "Blood delivery test one", "value": "blood_delivery_test_one"},
        {"label": "Blood delivery test two", "value": "blood_delivery_test_two"},
        {"label": "Blood delivery test three", "value": "blood_delivery_test_three"},
    ]

def get_layout(G, pos, labels):
    print("get_layout called - creating initial application layout.")
    return html.Div(
        className="dbc container-fluid vh-100",
        children=[
            dcc.Store(data='', id='session-id', storage_type='session'),
            dbc.Row(
                className="vh-100",
                children=[
                    # div containing the graph, should take half of the screen
                    dbc.Col(style={"border": "1px solid green"}, children=[
                        cyto.Cytoscape(
                            id="graph_presenter",
                            layout={"name": "preset"},
                            style={"width": "98%", "height": "98%"},
                            elements=[
                                         {"data": {"id": str(node), "label": "Nod " + str(i)},
                                          "position": {"x": pos[node][0], "y": pos[node][1]}} for i, node in
                                         enumerate(G.nodes())
                                     ] + [
                                         {"data": {"source": str(edge[0]), "target": str(edge[1]),
                                                   "edge_label": str(idx)}} for idx, edge in enumerate(G.edges())
                                     ],
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
                        )]),

                    dbc.Col(style={"border": "1px solid gray"}, children=[
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
                            html.Div(id="user-session-block", style={"display":"none"}, children=[
                                html.Div(className="row align-items-left mt-2 ml-2", children=[
                                    html.Div(className="col-sm-2 mt-1", children=[
                                        html.Span("Hello, "),
                                        html.Span(id='user-email-show', style={"fontWeight":"bold", "fontSize":"larger"}),
                                    ]),
                                    html.Div(className="col-sm-2", children=[
                                        html.Button("Log out", id='logout-button',
                                                    className="btn btn-warning"),
                                    ]),
                                    html.Div(className="col-sm-6", children=[
                                        html.Select(id='user-saved-problems', className="form-select", children = [
                                            html.Option(value=problem["value"], children=problem["label"]) for problem in get_saved_problems_list()
                                        ]),
                                    ]),
                                    html.Div(className="col-sm-2", children=[
                                        html.Button("Load problem", id='load-problem', className="btn btn-primary")
                                    ]),
                                ])
                            ]),
                            html.Div(id="login-error-block", style={"display":"none", "color":"red", "textAlign":"center"})
                        ]),

                        html.H4("Problem editor", className="bg-primary text-white p-2 mb-2 mt-2 text-center"),
                        html.Div(className="form",
                                  children=[
                                      html.Div([html.H4("Edge data", className="my-0 py-0")], className="form row align-items-left mt-2 g-1"),
                                      html.Div(className="form row align-items-left mt-2 g-1",
                                               children=[
                                                   html.Div(className="col-sm-2", children=[
                                                       html.Label("Source node", htmlFor="source-node-input", className="form-label"),
                                                       dcc.Input(id='source-node-input', type='text',
                                                                 placeholder="node id", className="form-control"),
                                                   ]),
                                                   html.Div(className="col-sm-2", children=[
                                                       html.Label("Target node", htmlFor="target-node-input", className="form-label"),
                                                       dcc.Input(id='target-node-input', type='text',
                                                                 placeholder="node id", className="form-control"),
                                                   ]),
                                                   html.Div(className="col-sm-3", children=[
                                                       html.Label("Operational cost", htmlFor="oper-cost-input", className="form-label"),
                                                       dcc.Input(id='oper-cost-input', type='text',
                                                                 placeholder="function of flow", className="form-control"),
                                                   ]),
                                                   html.Div(className="col-sm-3", children=[
                                                       html.Label("Waste cost", htmlFor="waste-discard-cost-input", className="form-label"),
                                                       dcc.Input(id='waste-discard-cost-input', type='text',
                                                                 placeholder="function of flow", className="form-control"),
                                                   ]),
                                                   html.Div(className="col-sm-2", children=[
                                                       html.Label("Risk cost", htmlFor="risk-cost-input",
                                                                  className="form-label"),
                                                       dcc.Input(id='risk-cost-input', type='text',
                                                                 placeholder="function of flow",
                                                                 className="form-control"),
                                                   ]),
                                               ]),
                                      html.Div(className="form row align-items-left mt-2 g-1",
                                               children=[
                                                   html.Div(className="col-auto", children=[
                                                       dbc.Button("Add Edge", id='add-edge-button',
                                                                  className="btn btn-primary"),
                                                   ]),

                                                   html.Div(className="col-auto", children=[
                                                       html.Button("Set edge parameters", id='set-edge-params-button',
                                                                   className="btn btn-info"),
                                                   ]),

                                                   html.Div(className="col-auto", children=[
                                                       dbc.Button("Remove Edge", id='remove-edge-button',
                                                                  className="btn btn-danger"),
                                                   ]),
                                               ]),
                                      html.Hr(),
                                      html.Div([html.H4("Node data", className="my-0 py-0")],
                                               className="form row align-items-left mt-2 g-1"),
                                      html.Hr(),
                                  ]),

                        html.Div(id='console-output'),
                    ])
                ])
        ])



@app.callback(Output('console-output', 'children'),
              [
                  Input('graph_presenter', 'tapEdgeData'),
                  Input('graph_presenter', 'tapNodeData')
              ],
              [
                  State('session-id', 'data')
              ]
              )
def onGraphElementClick(edgeData, nodeData, session_data):
    global active_edge, active_node

    context = dash.ctx.triggered
    print(f"Context: {context}")
    print(f"Edge data: {edgeData}")
    print(f"Node data: {nodeData}")
    print(f"Session data: {session_data}")

    if session_data is not None:
        email = cache.get(session_data)
        print(f"Email: {email}")

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

    elements = [elem for elem in elements if ((not 'source' in elem['data']) or (
        not (elem['data']['source'] == source and elem['data']['target'] == target)))]

    return elements

@app.callback(
    [
        # Output('login-form-block', 'style'),
        # Output('user-session-block', 'style'),
        # Output('user-email-show', 'children'),
        # Output('email-input', 'value'),
        # Output('password-input', 'value'),
        Output('login-error-block', 'children'),
        Output('login-error-block', 'style'),
        Output('session-id', 'data')
    ],
    [
        Input('login-button', 'n_clicks'),
        Input('logout-button', 'n_clicks')
    ],
    [
        State('email-input', 'value'),
        State('password-input', 'value'),
        State('session-id', 'data')
    ]
)
def login_callback(n_clicks_login, n_clicks_logout, email, password, session_data):

    if n_clicks_login  is None and n_clicks_logout is None:
        raise PreventUpdate

    context = dash.ctx.triggered
    error = None
    error_block_style = Patch()
    logged_in = False

    if context[0]['prop_id'] == 'login-button.n_clicks':
        # hash password and check if it matches the one in the database
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # users and their data are saved in local directory 'users_data'
        # each user has a separate file named after their email, which contains hashed password

        # ensure that directory exists
        if not os.path.exists("users_data"):
            os.mkdir("users_data")

        # check if user exists and if password matches
        if os.path.exists(f"users_data/{email}.txt"):
            with open(f"users_data/{email}.txt", "r") as f:
                if f.read() == hashed_password:
                    logged_in = True
                    print("Logged in successfully")
                else:
                    logged_in = False
                    error = "Wrong password!"
                    error_block_style['display'] = 'block'
                    print("Wrong password!")
        else:
            # create new user
            with open(f"users_data/{email}.txt", "w") as f:
                f.write(hashed_password)

            logged_in = True
            print("Created new user")

        if logged_in:
            # generate sid and save it in cache
            sid = str(uuid.uuid4())
            cache.set(sid, email, timeout=60 * 60 * 24 * 7)
            print(f"Generated session id: {sid}")

            return [error, error_block_style, sid]
        else:
            return [error, error_block_style, '']

    elif context[0]['prop_id'] == 'logout-button.n_clicks':
        print("Logged out")
        # show login form and hide logout form, does not update email and password
        return [error, error_block_style, '']

    raise PreventUpdate

@app.callback(
    [
        Output('login-form-block', 'style'),
        Output('user-session-block', 'style'),
        Output('user-email-show', 'children'),
        Output('email-input', 'value')
    ],
    [
        Input('session-id', 'data')
    ]
)
def session_changed(session_data):
    print(f"session_changed callback.")

    if session_data is not None and session_data != '':
        email = cache.get(session_data)
        print(f"Email in session: {email}")

        if email is not None:
            return [{"display": "none"}, {"display": "block"}, email, email]
        else:
            return [{"display": "block"}, {"display": "none"}, "", dash.no_update]
    else:
        return [{"display": "block"}, {"display": "none"}, "", dash.no_update]

if __name__ == "__main__":
    G, pos, labels = prepare_problem()
    app.layout = get_layout(G, pos, labels)

    cache.init_app(server)

    app.run_server(debug=True)