import hashlib
import json
import threading

import uuid
from wsgiref.simple_server import make_server

import flask
from flask_login import LoginManager
from flask_caching import Cache
import dash
from dash import clientside_callback
# import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import html, dcc, Patch
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import networkx as nx

from methods.algorithm_params import AlgorithmParams
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
from problems.testcases.blood_delivery import blood_delivery_hardcoded_test_one, blood_delivery_test_one, \
    blood_delivery_test_two, blood_delivery_test_three

import os
import logging
from logging.handlers import RotatingFileHandler

from net_editor_layout import get_layout, get_cytoscape_graph_elements, update_net_by_cytoscape_elements, \
    build_graph_view_layout

# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/cytoscape/events

# https://community.plotly.com/t/how-to-update-cytoscape-elements-list-of-dics-using-patch/75631

active_edge = None
active_node = None

# enum for cache keys
CACHE_KEY_PROBLEM = 'problem'
CACHE_KEY_ACTIVE_EDGE = 'active_edge'
CACHE_KEY_ACTIVE_NODE = 'active_node'
CACHE_KEY_EMAIL = 'email'

# dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
bs_css = ("https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css")

# Create the Dash application based on Flask server
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,
                title='VI algorithms test suite',
                update_title='Loading...',
                external_stylesheets=[bs_css]
                )
#                , external_stylesheets=["netedit.css"]

flask_cache_config = {
    'CACHE_TYPE': 'SimpleCache'
    # try 'FileSystemCache' if you don't want to setup redis
    # 'cache_type': 'redis',
    # 'cache_redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}

logger = logging.getLogger('vi-algo-test-suite')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

fh = RotatingFileHandler('vi-algo-test-suite.log', maxBytes=1000000, backupCount=100)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Starting server")

cache = Cache(config=flask_cache_config)

# Configure flask login with secret key from environment variable
server.config.update(SECRET_KEY=os.getenv('WEB_APP_SECRET_KEY'))

login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'


def get_cache_key(session_id, key):
    return f"{session_id}.{key}"


def set_cached_value(session_id, key, value, *, timeout=60 * 60 * 24):
    cache_key = get_cache_key(session_id, key)
    cache.set(cache_key, value, timeout=timeout)


def get_cached_value(session_id, key):
    cache_key = get_cache_key(session_id, key)
    return cache.get(cache_key)


def save_problem_to_cache(session_id, problem):
    set_cached_value(session_id, CACHE_KEY_PROBLEM, problem)


def get_problem_from_cache(session_id):
    return get_cached_value(session_id, CACHE_KEY_PROBLEM)


def prepare_default_problem():
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

    return problem


def get_initial_layout():
    session_id = str(uuid.uuid4())

    problem = prepare_default_problem()

    # calculate and update positions inside problem
    res = get_layout(problem, session_id)

    save_problem_to_cache(session_id, problem)

    return res


@app.callback(
    [
        Output('console-output', 'children', allow_duplicate=True),
        Output('source-node-input', 'value'),
        Output('target-node-input', 'value'),
        Output('selected-edge-index', 'value'),
        Output('oper-cost-input', 'value'),
        Output('oper-cost-deriv-input', 'value'),
        Output('waste-discard-cost-input', 'value'),
        Output('waste-discard-cost-deriv-input', 'value'),
        Output('risk-cost-input', 'value'),
        Output('risk-cost-deriv-input', 'value'),
        Output('edge-loss-input', 'value'),
    ],
    [
        Input({"type":"graph_presenter", "id": ALL}, 'tapEdgeData'),
        Input({"type":"graph_presenter", "id": ALL}, 'tapNodeData')
    ],
    [
        State('source-node-input', 'value'),
        State('target-node-input', 'value'),
        State({"type":"graph_presenter", "id": ALL}, 'elements'),
        State('session-id', 'data')
    ],
    prevent_initial_call=True
)
def onGraphElementClick(edgeDatas, nodeDatas, sourceNodes, targetNode, graphs_elements, session_id):

    # Only one graph presenter is used, so there should be only one element in the list
    if len(graphs_elements) != 1:
        logger.error(f"onGraphElementClick: len(graphs_elements) != 1: {len(graphs_elements)}")
        raise PreventUpdate

    edgeData = edgeDatas[0]
    nodeData = nodeDatas[0]
    sourceNode = sourceNodes[0] if sourceNodes is not None else None
    graph_elements = graphs_elements[0]

    problem = get_problem_from_cache(session_id)

    logger.info(f"onGraphElementClick: session_id: {session_id}; First node position: {graph_elements[0]['position']}")

    context = dash.ctx.triggered
    event_source = context[0]['prop_id']

    logger.info(f"onGraphElementClick event source: {event_source}")

    source_node_value = dash.no_update
    target_node_value = dash.no_update
    oper_cost_value = dash.no_update
    oper_cost_deriv_value = dash.no_update
    waste_discard_cost_value = dash.no_update
    waste_discard_cost_deriv_value = dash.no_update
    risk_cost_value = dash.no_update
    risk_cost_deriv_value = dash.no_update
    alpha_value = dash.no_update

    selected_edge_index = dash.no_update

    console_message = ''

    if event_source.endswith('.tapEdgeData') and edgeData:
        logger.info(f"onGraphElementClick: edge data: {edgeData}")
        set_cached_value(session_id, CACHE_KEY_ACTIVE_EDGE, edgeData)
        set_cached_value(session_id, CACHE_KEY_ACTIVE_NODE, None)

        source_node_value = edgeData['source']
        target_node_value = edgeData['target']
        selected_edge_index = int(edgeData['edge_index'])

        oper_cost_value, oper_cost_deriv_value = problem.net.c_string[selected_edge_index]
        waste_discard_cost_value, waste_discard_cost_deriv_value = problem.net.z_string[selected_edge_index]


        if problem.net.r_string and selected_edge_index < len(problem.net.r_string):
            risk_cost_value, risk_cost_deriv_value = problem.net.r_string[selected_edge_index]
        else:
            risk_cost_value = ''
            risk_cost_deriv_value = ''

        alpha_value = problem.net.edge_loss[selected_edge_index]

        console_message = "clicked/tapped the edge between " + edgeData['source'].upper() + " and " + edgeData[
            'target'].upper()
    elif event_source.endswith('.tapNodeData') and nodeData:
        logger.info(f"onGraphElementClick: node data: {nodeData}")

        for el in graph_elements:
            if el['data']['id'] == nodeData['id']:
                logger.info(f"onGraphElementClick: full node element data: {el}")
                break

        prev_active_node = get_cached_value(session_id, CACHE_KEY_ACTIVE_NODE)
        set_cached_value(session_id, CACHE_KEY_ACTIVE_NODE, el)
        set_cached_value(session_id, CACHE_KEY_ACTIVE_EDGE, None)

        if sourceNode and sourceNode != nodeData['id']:
            target_node_value = nodeData['id']
        elif sourceNode == nodeData['id']:
            source_node_value = None
            target_node_value = None
        else:
            source_node_value = nodeData['id']
            target_node_value = None

        console_message = "clicked/tapped the node " + nodeData['id'].upper() + '; First node pos: ' + str(
            graph_elements[0]['position']) + '; Sec: ' + str(graph_elements[1]['position'])

    return console_message, source_node_value, target_node_value, selected_edge_index, \
        oper_cost_value, oper_cost_deriv_value, waste_discard_cost_value, waste_discard_cost_deriv_value, \
        risk_cost_value, risk_cost_deriv_value, alpha_value


@app.callback(
    [
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
def login_callback(n_clicks_login, n_clicks_logout, email, password, session_id):
    logger.info(f"login_callback: session_id: {session_id}")

    if n_clicks_login is None and n_clicks_logout is None:
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
                    logger.info("Logged in successfully")
                else:
                    logged_in = False
                    error = "Wrong password!"
                    error_block_style['display'] = 'block'
                    logger.info("Wrong password!")
        else:
            # create new user
            with open(f"users_data/{email}.txt", "w") as f:
                f.write(hashed_password)

            logged_in = True
            logger.info("Created new user")

        if logged_in:
            # generate sid and save it in cache
            sid = str(uuid.uuid4())
            set_cached_value(sid, CACHE_KEY_EMAIL, email, timeout=60 * 60 * 24 * 7)
            logger.info(f"Generated session id: {sid}")

            if not os.path.exists(f"users_data/{email}"):
                os.mkdir(f"users_data/{email}")

            return [error, error_block_style, sid]
        else:
            return [error, error_block_style, '']

    elif context[0]['prop_id'] == 'logout-button.n_clicks':
        logger.info("Logged out")
        # show login form and hide logout form, does not update email and password
        return [error, error_block_style, '']

    raise PreventUpdate


@app.callback(
    [
        Output('login-form-block', 'style'),
        Output('user-session-block', 'style'),
        Output('user-email-show', 'children'),
        Output('email-input', 'value'),
        Output('user-saved-problems', 'options'),
        Output('session-id-input', 'value')
    ],
    [
        Input('session-id', 'data')
    ],
    [
        State('session-id-input', 'value')
    ]
)
def session_changed(new_session_id, old_session_id):
    logger.info(f"session_changed callback. storage session_id: {new_session_id}, input_session_id: {old_session_id}")

    problem = get_problem_from_cache(old_session_id)

    if new_session_id is not None and new_session_id != '':
        if new_session_id != old_session_id and problem is not None:
            save_problem_to_cache(new_session_id, problem)
            logger.info(f"Copied problem from old to new session {old_session_id} -> {new_session_id}")
        elif problem is None:
            problem = get_problem_from_cache(new_session_id)
            logger.info(
                f"There is no problem in old session {old_session_id}, but there is one in new session {new_session_id}")

        email = get_cached_value(new_session_id, CACHE_KEY_EMAIL)

        if email is not None:
            logger.info(f"Email in new session: {email}")
            problems = os.listdir(f"users_data/{email}")
            problem_dropdown_options = [{'value': problem, 'label': problem} for problem in problems]

            return [{"display": "none"}, {"display": "block"}, email, email, problem_dropdown_options, new_session_id]
        else:
            logger.info(f"No email in new session.")
            return [{"display": "block"}, {"display": "none"}, "", dash.no_update, [{'value': '', 'label': 'Default'}],
                    new_session_id]
    else:
        logger.info(f"New session is empty.")
        return [{"display": "block"}, {"display": "none"}, "", dash.no_update, [], dash.no_update]


clientside_callback(
    """
    function(data, elements) {
        console.log("clientside_callback:  " + data);
        console.log(elements);
        
        return 'ready';
    }
    """,
    Output('temp-data-target', 'value'),
    Input('temp-data', 'value'),
    State({"type":"graph_presenter", "id": ALL}, "elements")
)


# @app.callback(
#     Input('temp-data-target', 'value'),
#     State('graph_presenter', 'elements')
# )
# def update_graph_container_post_callback(temp_data, elements):
#     logger.info(f"update_graph_container_post_callback: {temp_data}")


# Changing elements broke the update cycle, positions are not updated after loading a problem
# https://github.com/plotly/dash-cytoscape/issues/159
@app.callback(
    Output('graph-container', 'children'),
    Output('save-problem-name-input', 'value'),
    Output('temp-data', 'value'),
    Input('load-problem-button', 'n_clicks'),
    State('user-saved-problems', 'value'),
    State('session-id', 'data'),
#    State('graph_presenter', 'elements'),
    prevent_initial_call=True
)
def load_problem_click(n_clicks, problem_name, session_id): # , elements
    if n_clicks is None or not session_id:
        raise PreventUpdate

    logger.info(f"load_problem_click: session_id: {session_id}, problem_name: {problem_name}")

    problem_dir_name = problem_name.replace(" ", "_")

    user_email = get_cached_value(session_id, CACHE_KEY_EMAIL)
    if user_email:
        problem_dir = f"users_data/{user_email}/{problem_dir_name}"
    else:
        problem_dir = f"users_data/{session_id}"

    if os.path.exists(problem_dir):
        problem = get_problem_from_cache(session_id)

        problem.net.loadFromDir(problem.net, path_to_load=problem_dir)
        logger.info(f"load_problem_click: problem loaded from file {problem_dir_name}")

        G, pos, labels = problem.net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0)

        if problem.net.pos:
            pos = problem.net.pos
            logger.info(f"load_problem_click: positions got from the problem.net structure.")
        else:
            logger.info(f"load_problem_click: positions calculated by to_nx_graph.")

        logger.info(f"load_problem_click positions: {pos}")

        new_graph_view = build_graph_view_layout(problem.net, G, pos, labels)
        # new_elements = get_cytoscape_graph_elements(problem.net, G=G, pos=pos, labels=labels)

        # elements_patch = Patch()
        # elements_patch[0]['position']['x'] = 10

        # for idx,old_el in enumerate(elements):
        #     # elements_patch.remove(old_el)
        #     if 'position' in new_elements[idx]:
        #         elements_patch[idx]['position']['x'] = new_elements[idx]['position']['x']
        #         elements_patch[idx]['position']['y'] = new_elements[idx]['position']['y']

        # elements_patch[idx]['data'] = new_elements[idx]['data']

        # elements_patch.extend(new_elements)

        # # # clear elements
        # elements.clear()
        # # add new elements
        # elements.extend(new_elements)

        save_problem_to_cache(session_id, problem)

        logger.info(f"load_problem_click: elements ready.")
        return new_graph_view, problem_name, 'NOT_USED' # json.dumps(new_elements)
    else:
        logger.info(f"Problem directory does not exist: {problem_dir_name}")
        return dash.no_update, dash.no_update, dash.no_update


# dash callback for click on button with id="set-edge-params-button"

@app.callback(
#    Output('graph-container', 'children', allow_duplicate=True),
    Output('console-output', 'children', allow_duplicate=True),
    Input('set-edge-params-button', 'n_clicks'),
    State('source-node-input', 'value'),
    State('target-node-input', 'value'),
    State('selected-edge-index', 'value'),
    State('oper-cost-input', 'value'),
    State('oper-cost-deriv-input', 'value'),
    State('waste-discard-cost-input', 'value'),
    State('waste-discard-cost-deriv-input', 'value'),
    State('risk-cost-input', 'value'),
    State('risk-cost-deriv-input', 'value'),
    State('edge-loss-input', 'value'),
    State('session-id', 'data'),
#    State({"type":"graph_presenter", "id": MATCH}, 'elements'),
    prevent_initial_call=True
)
def set_edge_params_click(
        n_clicks, source_node, target_node, selected_edge_index, oper_cost, oper_cost_deriv, waste_discard_cost,
        waste_discard_cost_deriv,
        risk_cost, risk_cost_deriv, edge_loss, session_id):

    if n_clicks is None:
        raise PreventUpdate

    selected_edge_index = int(selected_edge_index)

    logger.info(
        f"set_edge_params_click: called for session_id: {session_id}, source_node: {source_node}, target_node: {target_node}, selected_edge_index: {selected_edge_index}")

    problem = get_problem_from_cache(session_id)
    problem.net.c_string[selected_edge_index] = (oper_cost, oper_cost_deriv)
    problem.net.z_string[selected_edge_index] = (waste_discard_cost, waste_discard_cost_deriv)

    if risk_cost is not None and risk_cost != "":
        problem.net.r_string[selected_edge_index] = (risk_cost, risk_cost_deriv)

    if edge_loss is not None and edge_loss != "":
        problem.net.edge_loss[selected_edge_index] = edge_loss

    problem.net.update_functions_from_strings()

    save_problem_to_cache(session_id, problem)

    # do not need to update view?
    # G, pos, labels = problem.net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0)
    #
    # if problem.net.pos:
    #     pos = problem.net.pos
    #     logger.info(f"load_problem_click: positions got from the problem.net structure.")
    # else:
    #     logger.info(f"load_problem_click: positions calculated by to_nx_graph.")
    #
    # new_graph_view = build_graph_view_layout(problem.net, G, pos, labels)

    return f"Edge parameters updated for edge {source_node} -> {target_node}"


@app.callback(
    [
        Output('save-error-block', 'children'),
        Output('save-error-block', 'style')
    ],
    [
        Input('save-problem-button', 'n_clicks')
    ],
    [
        State('save-problem-name-input', 'value'),
        State({"type":"graph_presenter", "id": ALL}, 'elements'),
        State('session-id', 'data')
    ]
)
def save_problem_click(n_clicks, problem_name, graphs_elements, session_id):
    if n_clicks is None:
        raise PreventUpdate

    if not problem_name:
        return "Please enter problem name", {'color': 'red'}

    logger.info(f"save_problem_click: session_id: {session_id}, problem_name: {problem_name}")

    # Only one graph presenter is used, so there should be only one element in the list
    if len(graphs_elements) != 1:
        logger.error(f"save_problem_click: len(graphs_elements) != 1: {len(graphs_elements)}")
        raise PreventUpdate

    graph_elements = graphs_elements[0]

    for elem in graph_elements:
        if 'position' in elem:
            print(elem['data']['id'], (elem['position']['x'], elem['position']['y']))

    if session_id:
        problem_dir_name = problem_name.replace(" ", "_")

        problem = get_problem_from_cache(session_id)

        # update_net_by_cytoscape_elements(graph_elements, problem.net)
        # problem.net.update_functions_from_strings()

        user_email = get_cached_value(session_id, CACHE_KEY_EMAIL)

        if user_email:
            path_to_save = f"users_data/{user_email}/{problem_dir_name}"
            problem.saveToDir(path_to_save=path_to_save)
            logger.info(f"save_problem_click: problem saved to {path_to_save}")
            return "", {"display": "none"}
        else:
            logger.info(f"save_problem_click: not saved - not logged in")
            return "You need to log in to be able to save the problem setup!", {"display": "block"}
    else:
        logger.info(f"save_problem_click: not saved - no session and not logged in")
        return "You need to log in to be able to save the problem setup!", {"display": "block"}

if __name__ == "__main__":
    cache.init_app(server)

    app.layout = get_initial_layout

    app.run_server(debug=True)


# if __name__ == "__main__":
#     cache.init_app(server)
#
#     server = make_server("localhost", 8050, server)
#
#
#     def run_app():
#         app.layout = get_initial_layout
#         server.run(debug=True, use_reloader=False)
#
#
#     dash_thread = threading.Thread(target=run_app)
#     dash_thread.start()
#
#     # app.layout = get_initial_layout
#     # app.run_server(debug=True)
#
#
# @app.route('/test')
# def test():
#     return 'Hello World!'

#
# def stop_execution():
#     global keepPlot
#     # stream.stop_stream()
#     keepPlot = False
#     # stop the Flask server
#     server.shutdown()
#     server_thread.join()
#     print("Dash app stopped gracefully.")
#
#
# server = Flask(__name__)
# app = Dash(__name__, server=server)
#
# if __name__ == "__main__":
#     # create a server instance
#     server = make_server("localhost", 8050, server)
#     # start the server in a separate thread
#     server_thread = threading.Thread(target=server.serve_forever)
#     server_thread.start()
#
#
#     # start the Dash app in a separate thread
#     def start_dash_app():
#         app.run_server(debug=True, use_reloader=False)
#
#
#     dash_thread = threading.Thread(target=start_dash_app)
#     dash_thread.start()
#
#     while keepPlot:
#         time.sleep(1)  # keep the main thread alive while the other threads are running
