import dash
from dash import dcc
from dash import html
import networkx as nx
import plotly.graph_objs as go

from dash.dependencies import Input, Output, State

# Create the graph using NetworkX
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])

# Initialize node positions
pos = nx.spring_layout(G)

# Create the Plotly figure
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1, color="#888"),
    hoverinfo="none",
    mode="lines",
)

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[str(node) for node in G.nodes()],
    mode="markers+text",
    hoverinfo="text",
    marker=dict(
        showscale=False,
        colorscale="YlGnBu",
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title="Node Connections",
            xanchor="left",
            titleside="right",
        ),
        line_width=2,
    ),
)

# Create the Plotly layout
layout = go.Layout(
    title="Sample Graph",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
)

# Create the Dash application
app = dash.Dash(__name__)

# Define the Dash layout
app.layout = html.Div(
    children=[
        html.H1("Graph Visualization"),
        dcc.Graph(
            id="graph",
            figure=go.Figure(data=[edge_trace, node_trace], layout=layout),
            style={"width": "800px", "height": "600px"},
            config={"editable": True, "edits": {"shapePosition": True}}
        ),
        html.Button("Save", id="save-button"),
    ]
)


@app.callback(
    Output("graph", "figure"),
    Input("graph", "relayoutData"),
    State("graph", "figure"),
)
def update_positions(relayout_data, figure):
    if relayout_data is not None and "xaxis.range[0]" in relayout_data:
        # Update node positions based on relayout data
        new_pos = {
            node: [figure["data"][1]["x"][i], figure["data"][1]["y"][i]]
            for i, node in enumerate(G.nodes())
        }
        nx.set_node_attributes(G, new_pos, "pos")

    # Update node and edge positions
    node_trace["x"] = [pos[node][0] for node in G.nodes()]
    node_trace["y"] = [pos[node][1] for node in G.nodes()]
    edge_trace["x"] = []
    edge_trace["y"] = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # edge_trace["x"] += [x0, x1, None]
        edge_trace["x"] += tuple([x0, x1, None])

        # edge_trace["y"] += [y0, y1, None]
        edge_trace["y"] += tuple([y0, y1, None])

    return go.Figure(data=[edge_trace, node_trace], layout=layout)


@app.callback(
    Output("save-button", "disabled"),
    Input("graph", "relayoutData"),
)
def enable_save_button(relayout_data):
    if relayout_data is not None and "xaxis.range[0]" in relayout_data:
        return False
    return True


@app.callback(
    Output("save-button", "n_clicks"),
    Input("save-button", "n_clicks"),
    State("graph", "figure"),
)
def save_positions(n_clicks, figure):
    if n_clicks is not None:
        new_pos = {
            node: [figure["data"][1]["x"][i], figure["data"][1]["y"][i]]
            for i, node in enumerate(G.nodes())
        }
        nx.set_node_attributes(G, new_pos, "pos")
        # Perform further backend processing or save the updated positions

    return None


if __name__ == "__main__":
    app.run_server(debug=True)
