import hashlib
import json
import re
import base64
import mimetypes
from transliterate import translit
import threading

import dotenv

import uuid
from datetime import datetime
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
import diskcache
import networkx as nx
import numpy as np

from methods.algorithm_params import AlgorithmParams
from problems.blood_supply_net_problem import BloodSupplyNetwork, BloodSupplyNetworkProblem
from problems.testcases.blood_delivery import blood_delivery_hardcoded_test_one, blood_delivery_test_one, \
    blood_delivery_test_two, blood_delivery_test_three

import os
import logging
from logging.handlers import RotatingFileHandler

from net_editor_layout import get_layout, get_cytoscape_graph_elements, update_net_by_cytoscape_elements, \
    build_graph_view_layout
from run_algs_lib import AlgsRunner

# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/cytoscape/events

# https://community.plotly.com/t/how-to-update-cytoscape-elements-list-of-dics-using-patch/75631

dotenv.load_dotenv()

BASE_STORAGE_DIR = os.getenv('BASE_STORAGE_DIR')
USERS_DATA_DIR = os.path.join(BASE_STORAGE_DIR, 'web_users_data')
RUN_STATS_SUBDIR = 'alg_run_stats'
LOGS_DIR = os.path.join(BASE_STORAGE_DIR, 'web-logs')

os.makedirs(USERS_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

cbk_cache = diskcache.Cache("./cbk_cache")
background_callback_manager = dash.DiskcacheManager(cbk_cache)

active_edge = None
active_node = None

# enum for cache keys
CACHE_KEY_PROBLEM = 'problem'
CACHE_KEY_PARAMS = 'alg_params'
CACHE_KEY_ACTIVE_EDGE = 'active_edge'
CACHE_KEY_ACTIVE_NODE = 'active_node'
CACHE_KEY_EMAIL = 'email'

memory_cache = {}

# dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
bs_css = ("https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css")

# Create the Dash application based on Flask server
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,
                title='Програмний комплекс на базі алгоритмів для варіаційних нерівностей',
                background_callback_manager=background_callback_manager,
                update_title='Завантаження ...',
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

fh = RotatingFileHandler(os.path.join(LOGS_DIR, 'vi-algo-test-suite.log'), maxBytes=1000000, backupCount=100)
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


def replace_path_spec_chars(source: str, replace_char: str = '_') -> str:
    return re.sub("[^0-9a-zA-Z_]+", replace_char, source)


def get_cache_key(session_id, key):
    return f"{session_id}.{key}"


def set_cached_value(session_id, key, value, *, timeout=60 * 60 * 24):
    cache_key = get_cache_key(session_id, key)

    memory_cache[cache_key] = (value, datetime.utcnow(), timeout)
    # cache.set(cache_key, value, timeout=timeout)


def get_cached_value(session_id, key):
    cache_key = get_cache_key(session_id, key)

    if cache_key in memory_cache:
        return memory_cache[cache_key][0]
    else:
        return None

    # return cache.get(cache_key)


def get_user_folder(user_email):
    return os.path.join(USERS_DATA_DIR, replace_path_spec_chars(user_email)) if user_email else os.path.join(USERS_DATA_DIR, 'default')


def save_problem_to_cache(session_id, problem, alg_params=None):
    set_cached_value(session_id, CACHE_KEY_PROBLEM, problem)
    if alg_params:
        set_cached_value(session_id, CACHE_KEY_PARAMS, alg_params)


def get_problem_from_cache(session_id):
    return get_cached_value(session_id, CACHE_KEY_PROBLEM)


def get_params_from_cache(session_id):
    return get_cached_value(session_id, CACHE_KEY_PARAMS)


def prepare_default_problem():
    params_dict = {
        'eps': 1e-5,
        'min_iters': 10,
        'max_iters': 500,
        'lam': 0.01,
        'lam_KL': 0.005,
        'start_adaptive_lam': 0.5,
        'start_adaptive_lam1': 0.5,
        'adaptive_tau': 0.75,
        'adaptive_tau_small': 0.45,
        'show_plots': False,
        'save_history': True,
        'excel_history': True
    }

    algorithm_params = AlgorithmParams(**params_dict)

    problem = blood_delivery_test_three.prepareProblem(algorithm_params=algorithm_params, show_network=False,
                                                       print_data=False)

    return problem, algorithm_params


def get_initial_layout():
    session_id = str(uuid.uuid4())

    problem, algorithm_params = prepare_default_problem()

    # calculate and update positions inside problem
    res = get_layout(problem, session_id)

    save_problem_to_cache(session_id, problem, algorithm_params)

    return res


def build_current_graph_view(problem):
    G, pos, labels = problem.net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0)
    if problem.net.pos:
        pos = problem.net.pos
    return build_graph_view_layout(problem.net, G, pos, labels)


def sync_solver_dimensions(problem, alg_params=None):
    problem.rebuild_after_net_change()
    if alg_params is None:
        return

    arity = problem.net.n_p
    if not isinstance(alg_params.x0, np.ndarray) or alg_params.x0.shape != (arity,):
        alg_params.x0 = problem.x0.copy()

    if not isinstance(alg_params.x1, np.ndarray) or alg_params.x1.shape != (arity,):
        alg_params.x1 = alg_params.x0.copy()


def image_file_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/png'

    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('ascii')

    return f"data:{mime_type};base64,{encoded_image}"


def build_solver_image_preview(image_path, image_index):
    image_src = image_file_to_data_url(image_path)
    image_name = os.path.basename(image_path)

    return html.Button(
        id={"type": "solver-result-image", "index": image_index},
        n_clicks=0,
        title=f"Open {image_name}",
        className="btn p-0 border-0 bg-transparent w-100 mb-3",
        style={"cursor": "zoom-in"},
        children=html.Img(
            src=image_src,
            alt=image_name,
            style={
                "width": "100%",
                "border": "1px solid #dee2e6",
                "borderRadius": "4px",
                "backgroundColor": "white"
            }
        )
    )


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
        Output('risk-cost-input', 'disabled'),
        Output('risk-cost-deriv-input', 'disabled'),
        Output('edge-loss-input', 'value'),
        Output('expected_demand_min', 'value'),
        Output('expected_demand_max', 'value'),
        Output('expected_demand_distribution_type', 'value'),
        Output('expected_demand_min', 'disabled'),
        Output('expected_demand_max', 'disabled'),
        Output('expected_demand_distribution_type', 'disabled'),
        Output('selected-node-id', 'value'),
    ],
    [
        Input({"type": "graph_presenter", "id": ALL}, 'tapEdgeData'),
        Input({"type": "graph_presenter", "id": ALL}, 'tapNodeData')
    ],
    [
        State('source-node-input', 'value'),
        State('target-node-input', 'value'),
        State({"type": "graph_presenter", "id": ALL}, 'elements'),
        State('session-id', 'data')
    ],
    prevent_initial_call=True
)
def onGraphElementClick(edgeDatas, nodeDatas, sourceNodes, targetNodes, graphs_elements, session_id):
    # Only one graph presenter is used, so there should be only one element in the list
    if len(graphs_elements) != 1:
        logger.error(f"onGraphElementClick: len(graphs_elements) != 1: {len(graphs_elements)}")
        raise PreventUpdate

    edgeData = edgeDatas[0]
    nodeData = nodeDatas[0]
    sourceNode = sourceNodes if sourceNodes is not None else ''
    targetNode = targetNodes if targetNodes is not None else ''
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
    risk_cost_enabled = False

    expected_demand_min = dash.no_update
    expected_demand_max = dash.no_update
    expected_demand_distribution = dash.no_update
    expected_demand_enabled = False

    alpha_value = dash.no_update

    selected_edge_index = dash.no_update
    selected_node_id = dash.no_update

    console_message = ''

    if event_source.endswith('.tapEdgeData') and edgeData:
        logger.info(f"onGraphElementClick: edge data: {edgeData}")
        set_cached_value(session_id, CACHE_KEY_ACTIVE_EDGE, edgeData)
        set_cached_value(session_id, CACHE_KEY_ACTIVE_NODE, None)
        selected_node_id = ''

        source_node_value = edgeData['source']
        target_node_value = edgeData['target']
        selected_edge_index = int(edgeData['edge_index'])

        oper_cost_value, oper_cost_deriv_value = problem.net.c_string[selected_edge_index]
        waste_discard_cost_value, waste_discard_cost_deriv_value = problem.net.z_string[selected_edge_index]

        # only collection edges have risk cost, and in the current problem structure they are edges starting from the
        # node 0
        risk_cost_enabled = problem.net.is_collection_edge(selected_edge_index)
        if problem.net.r_string and risk_cost_enabled:
            risk_cost_value, risk_cost_deriv_value = problem.net.r_string[selected_edge_index]
        else:
            risk_cost_value = ''
            risk_cost_deriv_value = ''

        alpha_value = problem.net.edge_loss[selected_edge_index]

        is_expected_demand_edge = problem.net.is_demand_point_edge(selected_edge_index)
        expected_demand_enabled = is_expected_demand_edge

        if is_expected_demand_edge:
            demand_idx = problem.net.demand_points_dic[int(edgeData['target'])]
            if problem.net.expected_demand and demand_idx < len(problem.net.expected_demand):
                expected_demand = problem.net.expected_demand[demand_idx]
                expected_demand_min = expected_demand.get("min")
                expected_demand_max = expected_demand.get("max")
                expected_demand_distribution = expected_demand.get("distribution", "uniform")
            else:
                expected_demand_min = ''
                expected_demand_max = ''
                expected_demand_distribution = 'uniform'

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
        clicked_node = nodeData['id']
        selected_node_id = clicked_node

        if sourceNode == clicked_node:
            source_node_value = ''
            target_node_value = ''
            selected_node_id = ''
            console_message = f"edge endpoint selection cleared from node {clicked_node}"
        elif targetNode == clicked_node:
            source_node_value = dash.no_update
            target_node_value = ''
            console_message = f"target node {clicked_node} cleared"
        elif sourceNode and targetNode:
            source_node_value = clicked_node
            target_node_value = ''
            console_message = f"started a new edge selection from node {clicked_node}"
        elif sourceNode:
            target_node_value = clicked_node
            console_message = f"selected target node {clicked_node}"
        else:
            source_node_value = clicked_node
            target_node_value = ''
            console_message = f"selected source node {clicked_node}"

    return console_message, source_node_value, target_node_value, selected_edge_index, \
        oper_cost_value, oper_cost_deriv_value, waste_discard_cost_value, waste_discard_cost_deriv_value, \
        risk_cost_value, risk_cost_deriv_value, not risk_cost_enabled, not risk_cost_enabled, alpha_value, \
        expected_demand_min, expected_demand_max, expected_demand_distribution, not expected_demand_enabled, \
        not expected_demand_enabled, not expected_demand_enabled, selected_node_id


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
    logger.info(f"login/logout callback: session_id: {session_id}")

    if n_clicks_login is None and n_clicks_logout is None:
        raise PreventUpdate

    context = dash.ctx.triggered
    error = None
    error_block_style = Patch()
    logged_in = False

    if context[0]['prop_id'] == 'login-button.n_clicks':

        email = email.strip()
        if not email or len(email) < 3:
            error = "Please enter email or username of at least 3 characters long!"
            error_block_style['display'] = 'block'
            logger.info("No email!")
            return [error, error_block_style, dash.no_update]

        password = password.strip()
        if not password or len(password) < 3:
            error = "Password must be at least 3 characters long!"
            error_block_style['display'] = 'block'
            logger.info("Password too short!")
            return [error, error_block_style, dash.no_update]

        # hash password and check if it matches the one in the database
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # users and their data are saved in the filesystem under USERS_DATA_DIR/{email} folder (spec. chars replaced)
        # hashed password is saved in USERS_DATA_DIR/{email}/hash.txt file

        user_data_path = get_user_folder(email)
        password_hash_file = os.path.join(user_data_path, "hash.txt")

        # ensure that directory USERS_DATA_DIR exists
        os.makedirs(user_data_path, exist_ok=True)

        # check if user exists and if password matches
        if os.path.exists(password_hash_file):
            with open(password_hash_file, "r") as f:
                if f.read() == hashed_password:
                    logged_in = True
                    logger.info("User logged in successfully")
                else:
                    logged_in = False
                    error = "Wrong password!"
                    error_block_style['display'] = 'block'
                    logger.info("Wrong password!")
        else:
            # create new user
            with open(password_hash_file, "w") as f:
                f.write(hashed_password)

            logged_in = True
            logger.info("Created new user")

        if logged_in:
            # generate sid and save it in cache
            sid = str(uuid.uuid4())
            set_cached_value(sid, CACHE_KEY_EMAIL, email, timeout=60 * 60 * 24 * 7)
            logger.info(f"Generated session id: {sid}")

            return [error, error_block_style, sid]
        else:
            return [error, error_block_style, '']

    elif context[0]['prop_id'] == 'logout-button.n_clicks':
        if session_id:
            set_cached_value(session_id, CACHE_KEY_EMAIL, None)
            logger.info("Logged out")
        else:
            logger.warning("Logout attempt with no session!")

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
    alg_params = get_params_from_cache(old_session_id)

    if new_session_id is not None and new_session_id != '':
        if new_session_id != old_session_id and problem is not None:
            save_problem_to_cache(new_session_id, problem, alg_params)
            logger.info(f"Copied problem and params from old to new session {old_session_id} -> {new_session_id}")
        elif problem is None:
            problem = get_problem_from_cache(new_session_id)
            logger.info(
                f"There is no problem in old session {old_session_id}, but there is one in new session {new_session_id}")

        email = get_cached_value(new_session_id, CACHE_KEY_EMAIL)

        if email is not None:
            logger.info(f"Email in new session: {email}")

            problems = os.listdir(get_user_folder(email))
            problems = [p for p in problems if os.path.isdir(os.path.join(get_user_folder(email), p))]

            problem_dropdown_options = [{'value': problem, 'label': problem} for problem in problems]

            return [{"display": "none"}, {"display": "block"}, email, email, problem_dropdown_options, new_session_id]
        else:
            logger.info(f"No email in the new session.")
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
    State({"type": "graph_presenter", "id": ALL}, "elements")
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
def load_problem_click(n_clicks, problem_name, session_id):  # , elements
    if n_clicks is None or not session_id:
        raise PreventUpdate

    logger.info(f"load_problem_click: session_id: %s, problem_name: %s", session_id, problem_name)

    latin_problem_name = problem_name

    try:
        latin_problem_name = translit(latin_problem_name, reversed=True)
    except:
        logger.warning("Could not transliterate problem name: %s", problem_name)

    problem_dir_name = replace_path_spec_chars(latin_problem_name)

    user_email = get_cached_value(session_id, CACHE_KEY_EMAIL)

    problem_dir = os.path.join(get_user_folder(user_email), problem_dir_name)

    logger.info("load_problem_click: expected problem dir: %s", problem_dir)

    if os.path.exists(problem_dir):
        problem = get_problem_from_cache(session_id)

        problem.net.loadFromDir(problem.net, path_to_load=problem_dir)
        alg_params = get_params_from_cache(session_id)
        sync_solver_dimensions(problem, alg_params)
        logger.info("load_problem_click: problem network data loaded from folder %s", problem_dir)

        G, pos, labels = problem.net.to_nx_graph(x_left=200, x_right=600, y_bottom=500, y_top=0)

        if problem.net.pos:
            pos = problem.net.pos
            logger.info("load_problem_click: positions got from the problem.net structure.")
        else:
            logger.info("load_problem_click: positions calculated by to_nx_graph.")

        logger.info("load_problem_click positions: %s", pos)

        new_graph_view = build_graph_view_layout(problem.net, G, pos, labels)

        save_problem_to_cache(session_id, problem, alg_params)

        logger.info("load_problem_click: elements ready.")
        return new_graph_view, problem_name, 'NOT_USED'  # json.dumps(new_elements)
    else:
        logger.warning("Problem directory does not exist: %s", problem_dir_name)
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
    State('expected_demand_min', 'value'),
    State('expected_demand_max', 'value'),
    State('expected_demand_distribution_type', 'value'),
    State('session-id', 'data'),
    #    State({"type":"graph_presenter", "id": MATCH}, 'elements'),
    prevent_initial_call=True
)
def set_edge_params_click(
        n_clicks, source_node, target_node, selected_edge_index, oper_cost, oper_cost_deriv, waste_discard_cost,
        waste_discard_cost_deriv,
        risk_cost, risk_cost_deriv, edge_loss, expected_demand_min, expected_demand_max,
        expected_demand_distribution, session_id):
    if n_clicks is None:
        raise PreventUpdate

    selected_edge_index = int(selected_edge_index)

    logger.info(
        f"set_edge_params_click: called for session_id: {session_id}, source_node: {source_node}, target_node: {target_node}, selected_edge_index: {selected_edge_index}")

    problem = get_problem_from_cache(session_id)
    problem.net.c_string[selected_edge_index] = (oper_cost, oper_cost_deriv)
    problem.net.z_string[selected_edge_index] = (waste_discard_cost, waste_discard_cost_deriv)

    if risk_cost is not None and risk_cost != "":
        while problem.net.r_string is not None and len(problem.net.r_string) <= selected_edge_index:
            problem.net.r_string.append(("0", "0"))
        problem.net.r_string[selected_edge_index] = (risk_cost, risk_cost_deriv)

    if edge_loss is not None and edge_loss != "":
        problem.net.edge_loss[selected_edge_index] = float(edge_loss)

    if problem.net.is_demand_point_edge(selected_edge_index):
        target_node = int(problem.net.edges[selected_edge_index][1])
        demand_idx = problem.net.demand_points_dic[target_node]
        if problem.net.expected_demand is None:
            problem.net.expected_demand = [
                problem.net.get_default_expected_demand() for _ in range(problem.net.n_R)
            ]

        default_spec = problem.net.get_default_expected_demand()
        spec = {
            "min": float(expected_demand_min) if expected_demand_min not in (None, '') else default_spec["min"],
            "max": float(expected_demand_max) if expected_demand_max not in (None, '') else default_spec["max"],
            "distribution": expected_demand_distribution or "uniform"
        }
        problem.net.expected_demand[demand_idx] = spec
        problem.net.expected_shortage, problem.net.expected_surplus = (
            problem.net.build_expected_functions_from_specs(problem.net.expected_demand)
        )

    problem.net.update_functions_from_strings()
    problem.net.rebuild_after_topology_change()
    problem.rebuild_after_net_change()

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
        Output('graph-container', 'children', allow_duplicate=True),
        Output('console-output', 'children', allow_duplicate=True),
        Output('source-node-input', 'value', allow_duplicate=True),
        Output('target-node-input', 'value', allow_duplicate=True),
        Output('selected-edge-index', 'value', allow_duplicate=True),
        Output('selected-node-id', 'value', allow_duplicate=True),
    ],
    [
        Input('add-vertex-button', 'n_clicks'),
        Input('remove-vertex-button', 'n_clicks'),
        Input('add-edge-button', 'n_clicks'),
        Input('remove-edge-button', 'n_clicks'),
        Input('clear-selection-button', 'n_clicks'),
    ],
    [
        State('add-vertex-layer-dropdown', 'value'),
        State('selected-node-id', 'value'),
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
    ],
    prevent_initial_call=True
)
def edit_topology_click(
        add_vertex_clicks, remove_vertex_clicks, add_edge_clicks, remove_edge_clicks, clear_selection_clicks,
        vertex_layer, selected_node_id, source_node, target_node, selected_edge_index, oper_cost, oper_cost_deriv,
        waste_discard_cost, waste_discard_cost_deriv, risk_cost, risk_cost_deriv, edge_loss, session_id):
    context = dash.ctx.triggered
    if not context or not session_id:
        raise PreventUpdate

    problem = get_problem_from_cache(session_id)
    if problem is None:
        raise PreventUpdate

    event_source = context[0]['prop_id']

    try:
        if event_source == 'add-vertex-button.n_clicks':
            new_node_id = problem.net.add_vertex(vertex_layer)
            message = f"Vertex {new_node_id} added to layer {vertex_layer}"
            source_node = str(new_node_id)
            target_node = ''
            selected_edge_index = ''
            selected_node_id = str(new_node_id)
        elif event_source == 'remove-vertex-button.n_clicks':
            node_to_remove = selected_node_id or source_node
            if node_to_remove in (None, ''):
                return dash.no_update, "Select a vertex before removing it.", dash.no_update, dash.no_update, \
                    dash.no_update, dash.no_update

            problem.net.remove_vertex(int(node_to_remove))
            message = f"Vertex {node_to_remove} removed"
            source_node = ''
            target_node = ''
            selected_edge_index = ''
            selected_node_id = ''
        elif event_source == 'add-edge-button.n_clicks':
            if source_node in (None, '') or target_node in (None, ''):
                return dash.no_update, "Select source and target vertices before adding an edge.", dash.no_update, \
                    dash.no_update, dash.no_update, dash.no_update

            default_c, default_z, default_r, default_loss = problem.net.get_default_edge_cost_strings()
            c_string = (oper_cost or default_c[0], oper_cost_deriv or default_c[1])
            z_string = (waste_discard_cost or default_z[0], waste_discard_cost_deriv or default_z[1])
            r_string = (risk_cost or default_r[0], risk_cost_deriv or default_r[1])
            new_edge_index = problem.net.add_edge(
                int(source_node), int(target_node),
                c_string=c_string,
                z_string=z_string,
                r_string=r_string,
                edge_loss=float(edge_loss) if edge_loss not in (None, '') else default_loss
            )
            message = f"Edge {new_edge_index} added: {source_node} -> {target_node}"
            selected_edge_index = str(new_edge_index)
            selected_node_id = ''
        elif event_source == 'remove-edge-button.n_clicks':
            if selected_edge_index in (None, ''):
                return dash.no_update, "Select an edge before removing it.", dash.no_update, dash.no_update, \
                    dash.no_update, dash.no_update

            problem.net.remove_edge(int(selected_edge_index))
            message = f"Edge {selected_edge_index} removed"
            selected_edge_index = ''
        elif event_source == 'clear-selection-button.n_clicks':
            return dash.no_update, "Selection cleared", '', '', '', ''
        else:
            raise PreventUpdate
    except ValueError as exc:
        logger.warning("edit_topology_click: %s", exc)
        return dash.no_update, str(exc), dash.no_update, dash.no_update, dash.no_update, dash.no_update

    problem.rebuild_after_net_change()
    save_problem_to_cache(session_id, problem)

    return build_current_graph_view(problem), message, source_node, target_node, selected_edge_index, selected_node_id


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
        State({"type": "graph_presenter", "id": ALL}, 'elements'),
        State('session-id', 'data')
    ]
)
def save_problem_click(n_clicks, problem_name, graphs_elements, session_id):
    if n_clicks is None:
        raise PreventUpdate

    problem_name = problem_name.strip() if problem_name else ''

    if not problem_name:
        return "Please enter problem name", {'color': 'red'}

    logger.info("save_problem_click: session_id: %s, problem_name: %s", session_id, problem_name)

    problem_name_latin = problem_name

    try:
        problem_name_latin = translit(problem_name, reversed=True)
    except:
        logger.warning("Could not transliterate problem name: %s", problem_name)

    problem_dir_name = replace_path_spec_chars(problem_name_latin)

    # Only one graph presenter is used, so there should be only one element in the list
    if len(graphs_elements) != 1:
        logger.error(f"save_problem_click: len(graphs_elements) != 1: {len(graphs_elements)}")
        raise PreventUpdate

    if session_id:
        problem = get_problem_from_cache(session_id)

        graph_elements = graphs_elements[0]
        update_net_by_cytoscape_elements(graph_elements, problem.net)
        problem.net.update_functions_from_strings()
        problem.net.rebuild_after_topology_change()
        problem.rebuild_after_net_change()

        user_email = get_cached_value(session_id, CACHE_KEY_EMAIL)

        if user_email:
            path_to_save = os.path.join(get_user_folder(user_email), problem_dir_name)
            problem.saveToDir(path_to_save=path_to_save)
            logger.info("save_problem_click: problem saved to %s", path_to_save)
            return "", {"display": "none"}
        else:
            logger.info("save_problem_click: not saved - not logged in")
            return "You need to log in to be able to save the problem setup!", {"display": "block"}
    else:
        logger.info(f"save_problem_click: not saved - no session and not logged in")
        return "You need to log in to be able to save the problem setup!", {"display": "block"}


@app.callback(
    [
        Output('solver-console-output', 'children'),
        Output('solver-images-output', 'children'),
    ],
    [
        Input('solve-problem-button', 'n_clicks')
    ],
    [
        State('solver-methods', 'value'),
        State('session-id', 'data')
    ],
    running=[
        (Output("solve-problem-button", "disabled"), True, False),
    ],
    background=True,
    prevent_initial_call=True
)
def solve_problem_click(n_clicks, solvers, session_id):
    logger.info("solve_problem_click: session_id: %s, solvers: %s", session_id, solvers)
    if n_clicks is None:
        raise PreventUpdate

    if session_id:
        alg_params = get_params_from_cache(session_id)
        problem = get_problem_from_cache(session_id)
        user_email = get_cached_value(session_id, CACHE_KEY_EMAIL)

        if user_email:
            sync_solver_dimensions(problem, alg_params)
            save_problem_to_cache(session_id, problem, alg_params)

            runner = AlgsRunner(problem=problem, params=alg_params,
                                runs_data_save_path=os.path.join(get_user_folder(user_email), RUN_STATS_SUBDIR))

            runner.init_algs()
            result = runner.run_algs(solvers)

            logger.info("solve_problem_click: solvers run completed. Data saved to %s", runner.runs_data_save_path)

            if result and 'run_log_file_path' in result:
                # read log file and return it as a result
                with open(result['run_log_file_path'], 'r') as f:
                    log_data = f.readlines()

                res_text = [html.Pre("\n".join(log_data))]
                res_images = []

                if 'graph_results' in result:
                    for image_index, image_path in enumerate(result['graph_results']):
                        res_images.append(build_solver_image_preview(image_path, image_index))

            return [res_text, res_images]
        else:
            logger.info("solve_problem_click: not saved - not logged in")
            return [["You need to log in to be able to test solvers!"], []]
    else:
        logger.info("solve_problem_click: no session and not logged in")
        return [["You need to log in to be able to test solvers!"], []]


@app.callback(
    [
        Output('solver-image-modal', 'style'),
        Output('solver-image-modal-img', 'src'),
        Output('solver-image-modal-img', 'alt'),
    ],
    [
        Input({"type": "solver-result-image", "index": ALL}, 'n_clicks'),
        Input('solver-image-modal-close', 'n_clicks'),
        Input('solver-image-modal-backdrop', 'n_clicks'),
    ],
    [
        State({"type": "solver-result-image", "index": ALL}, 'children'),
    ],
    prevent_initial_call=True
)
def solver_image_modal_click(image_clicks, close_clicks, backdrop_clicks, image_buttons):
    context = dash.ctx.triggered_id

    hidden_style = {"display": "none"}
    visible_style = {"display": "block"}

    if context in ('solver-image-modal-close', 'solver-image-modal-backdrop'):
        return hidden_style, dash.no_update, dash.no_update

    if not isinstance(context, dict) or context.get("type") != "solver-result-image":
        raise PreventUpdate

    image_index = context.get("index")
    if image_index is None or image_index >= len(image_buttons):
        raise PreventUpdate

    image_child = image_buttons[image_index]
    if isinstance(image_child, list):
        image_child = image_child[0] if image_child else None

    image_props = image_child.get("props", {}) if isinstance(image_child, dict) else {}
    return visible_style, image_props.get("src"), image_props.get("alt", "Solver plot")


if __name__ == "__main__":
    cache.init_app(server)

    app.layout = get_initial_layout

    app.run_server(debug=True)
