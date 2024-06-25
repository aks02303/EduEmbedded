import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import Axes3D module

def plot_tsne_3d(entities_csv, embeddings_csv, output_path):
    # Load the entity data
    entity_df = pd.read_csv(entities_csv, header=None, names=['entity'])

    # Load the entity embedding data
    embedding_df = pd.read_csv(embeddings_csv, header=None, names=[f'feature_{i}' for i in range(1, 41)])
    embedding_df['entity'] = entity_df['entity']

    # Assign colors based on prefixes
    colors = ['red' if entity.startswith('Course') else 'green' if entity.startswith('vi') else 'blue' if entity.startswith ('topic') else 'black' for entity in entity_df['entity']]

    # Perform t-SNE with 3 dimensions
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_tsne = tsne.fit_transform(embedding_df.iloc[:, :-1])  # Exclude the 'entity' column from embeddings

    # Plot the t-SNE visualization with different colors in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Scatter plot in 3D
    ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2], c=colors, alpha=0.5)

    # Create custom legend
    ax.scatter([], [], [], color='red', label='Courses')
    ax.scatter([], [], [], color='green', label='Vocabulary')
    ax.scatter([], [], [], color='blue', label='Topics')

    # Set labels and legend
    ax.set_title('t-SNE Visualization for All Entities (3D)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.legend()

    plt.savefig(output_path + '/tsne_3d.png')
    plt.show()

def plot_tsne_2d(entities_csv, embeddings_csv, output_path):
    # Load the entity data
    entity_df = pd.read_csv(entities_csv, header=None, names=['entity'])

    # Load the entity embedding data
    embedding_df = pd.read_csv(embeddings_csv, header=None, names=[f'feature_{i}' for i in range(1, 41)])
    embedding_df['entity'] = entity_df['entity']

    # Assign colors based on prefixes
    colors = ['red' if entity.startswith('Course') else 'green' if entity.startswith('vi') else 'blue' if entity.startswith('topic') else 'white' for entity in entity_df['entity']]

    # Perform t-SNE with 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embedding_df.iloc[:, :-1])  # Exclude the 'entity' column from embeddings

    # Plot the t-SNE visualization with different colors
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], color=colors, alpha=0.5)

    # Create custom legend
    plt.scatter([], [], color='red', label='Courses')
    plt.scatter([], [], color='green', label='Vocabulary')
    plt.scatter([], [], color='blue', label='Topics')

    plt.title('t-SNE Visualization for All Entities (2D)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()

    plt.savefig(output_path + '/tsne_2d.png')
    plt.show()

# Example usage:
plot_tsne_3d('D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entities.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entity_embeddings.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/Visualization/result')

plot_tsne_2d('D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entities.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entity_embeddings.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/Visualization/result')
