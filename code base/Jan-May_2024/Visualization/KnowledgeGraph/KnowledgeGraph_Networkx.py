from pyvis.network import Network
import random

def generate_graph_with_neighbors(nodes, data):
    # Filter data to include only the specified nodes and their neighbors
    node_data = data[data['head'].isin(nodes) | data['value'].isin(nodes)]
    neighbor_nodes = set(node_data['head']).union(set(node_data['value']))

    # Create a Pyvis Network instance
    net = Network(notebook=True)

    # Add nodes with random colors
    for node in neighbor_nodes:
        node_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Generate a random hex color code
        net.add_node(node, color=node_color)

    # Add edges
    for _, row in node_data.iterrows():
        net.add_edge(row['head'], row['value'], title=row['variable'])

    # Configure physics and interaction options
    net.options.physics.enabled = True
    net.options.interaction.hover = True
    net.options.interaction.zoomView = True
    net.options.interaction.dragNodes = True
    net.options.interaction.dragView = True
    net.options.interaction.showNodeNamesOnHover = True
    
    return net

# Example usage:
# Suppose you have nodes and data like this:
nodes = ['A', 'B', 'C']
data = {
    'head': ['A', 'A', 'B'],
    'value': ['B', 'C', 'C'],
    'variable': ['relation1', 'relation2', 'relation3']
}

# Generate and visualize the graph
graph = generate_graph_with_neighbors(nodes, data)
graph.show('graph.html')
