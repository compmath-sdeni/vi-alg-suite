import dash
import dash_cytoscape as cyto
from dash import html
from dash.dependencies import Input, Output, State
import networkx as nx

# https://dash.plotly.com/basic-callbacks
# https://dash.plotly.com/cytoscape/events

# Create the graph using NetworkX
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4, 5])
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

# Initialize node positions
pos = nx.spring_layout(G)
print(pos)
pos = [(100,100), (200, 100), (100,200), (200,200), (100, 300), (200, 300)]

# Create the Dash application
app = dash.Dash(__name__)

# Define the Dash layout
app.layout = html.Div(
    children=[
        html.H1("Graph Visualization"),
        cyto.Cytoscape(
            id="graph_presenter",
            layout={"name": "preset"},
            style={"width": "800px", "height": "800px", "border": "2px solid green"},
            elements=[
                         {"data": {"id": str(node), "label": "Nod "+str(i)}, "position": {"x": pos[node][0], "y": pos[node][1]}} for i, node in enumerate(G.nodes())
                     ] + [
                         {"data": {"source": str(edge[0]), "target": str(edge[1])}} for edge in G.edges()
                     ],
            stylesheet=[
                {
                    "selector" : 'node',
                    "style" : {
                        'background-color' : '#038',
                        'label' : 'data(id)'
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
        ),

        html.Div(id='console-output1'),
        html.Div(id='console-output2'),
        html.Div(),
        html.Button("Save", id="save-button", disabled=True)
    ]
)


@app.callback(Output('console-output1', 'children'),
              Input('graph_presenter', 'tapNodeData'))
def displayTapNodeData(data):
    if data:
        return "You recently clicked/tapped the city: " + data['label']


@app.callback(Output('console-output2', 'children'),
              Input('graph_presenter', 'tapEdgeData'))
def displayTapEdgeData(data):
    if data:
        return "You recently clicked/tapped the edge between " + \
               data['source'].upper() + " and " + data['target'].upper()

@app.callback(
    Output("save-button", "disabled"),
    Input("graph_presenter", "elements"),
)
def enable_save_button(elements):
    if elements:
        return False
    return True


if __name__ == "__main__":
    app.run_server(debug=True)
