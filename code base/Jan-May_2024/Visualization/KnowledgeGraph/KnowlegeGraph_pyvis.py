from pyvis.network import Network
import random
import pandas as pd

def generate_graph_with_neighbors(nodes, data):
    # Convert data to DataFrame
    data_df = pd.DataFrame(data, columns=['head', 'variable', 'value'])
    
    # Filter data to include only the specified nodes and their neighbors
    node_data = data_df[data_df['head'].isin(nodes) | data_df['value'].isin(nodes)]
    neighbor_nodes = set(node_data['head']).union(set(node_data['value']))

    # Create a Pyvis Network instance
    net = Network(notebook=True, height='100vh')  # Set graph height to 100vh

    # Add nodes with random colors
    for node in neighbor_nodes:
        node_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Generate a random hex color code
        net.add_node(node, color=node_color)

    # Add edges with relation information
    for _, row in node_data.iterrows():
        net.add_edge(row['head'], row['value'], title=row['variable'], label='', hover_title=row['variable'])

    # Configure physics and interaction options
    net.options.physics.enabled = True
    net.options.physics.barnesHut = {"gravitationalConstant": -2000, "springLength": 150, "springConstant": 0.03}
    net.options.physics.maxVelocity = 50
    net.options.physics.minVelocity = 0.1
    net.options.physics.solver = 'forceAtlas2Based'
    
    # Configure interaction options
    net.options.interaction.hover = True
    net.options.interaction.zoomView = True
    net.options.interaction.dragNodes = True
    net.options.interaction.dragView = True
    net.options.interaction.showNodeNamesOnHover = True
    net.options.interaction.tooltipDelay = 200  # Adjust tooltip delay
    
    return net

# Read data from CSV file
file_path = r"D:\EduEmbedd\code base\Jan-May_2024\output\triples.csv"
data = pd.read_csv(file_path)

# Extract nodes from data
nodes = set(data['head']).union(set(data['value']))

# Generate and visualize the graph
graph = generate_graph_with_neighbors(nodes, data)
graph.show('graph.html')
