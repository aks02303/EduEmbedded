import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import os

def plot_tsne_2d(entities_csv, embeddings_csv, similar_entities_csv, output_dir):
    # Load the entity data
    entity_df = pd.read_csv(entities_csv, header=None, names=['entity'])
    entity_df.columns = ['entity']

    # Load the entity embedding data
    embedding_df = pd.read_csv(embeddings_csv, header=None, names=[f'feature_{i}' for i in range(1, 41)])
    embedding_df['entity'] = entity_df['entity']

    # Load the similar entities data
    similar_entities_df = pd.read_csv(similar_entities_csv)

    # Perform t-SNE with 2 dimensions
    merged_df = pd.merge(entity_df, embedding_df, left_on='entity', right_on='entity')
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(merged_df.iloc[:, 1:])

    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(embeddings_tsne, columns=['tsne1', 'tsne2'])
    tsne_result_df = pd.concat([entity_df, tsne_df], axis=1)

    # Create a colormap to map each unique pair to a unique color
    pairs = similar_entities_df[['entity1', 'entity2']].values.tolist()
    unique_pairs = list(set(tuple(sorted(pair)) for pair in pairs))
    num_unique_pairs = len(unique_pairs)
    color_map = ListedColormap(plt.cm.tab10(np.linspace(0, 1, num_unique_pairs)))

    # Plot the t-SNE visualization for each pair with unique colors
    plt.figure(figsize=(12, 8))
    legend_patches = []  # List to store legend patches
    for index, (entity1, entity2) in enumerate(unique_pairs):
        color = color_map(index)

        x1, y1 = tsne_result_df.loc[tsne_result_df['entity'] == entity1, ['tsne1', 'tsne2']].values[0]
        x2, y2 = tsne_result_df.loc[tsne_result_df['entity'] == entity2, ['tsne1', 'tsne2']].values[0]

        # Plot nodes with the same color for both node and edge
        plt.scatter(x1, y1, color=color)
        plt.scatter(x2, y2, color=color)

        # Draw a line connecting the entities with the same color as nodes
        plt.plot([x1, x2], [y1, y2], color=color, alpha=0.5)

        # Create a legend patch for each pair
        legend_patches.append(Patch(color=color, label=f'{entity1}-{entity2}'))

    # Add legend
    # plt.legend(handles=legend_patches, loc='upper right')

    plt.title('t-SNE Visualization for Each Pair with Unique Colors (2D)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'tsne_2d_unique.png')
    plt.savefig(output_path)
    
    plt.show()

def plot_tsne_3d(entities_csv, embeddings_csv, similar_entities_csv, output_dir):
    # Load the entity data
    entity_df = pd.read_csv(entities_csv, header=None, names=['entity'])
    entity_df.columns = ['entity']

    # Load the entity embedding data
    embedding_df = pd.read_csv(embeddings_csv, header=None, names=[f'feature_{i}' for i in range(1, 41)])
    embedding_df['entity'] = entity_df['entity']

    # Load the similar entities data
    similar_entities_df = pd.read_csv(similar_entities_csv)

    # Perform t-SNE with 3 dimensions
    merged_df = pd.merge(entity_df, embedding_df, left_on='entity', right_on='entity')
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_tsne = tsne.fit_transform(merged_df.iloc[:, 1:])

    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(embeddings_tsne, columns=['tsne1', 'tsne2', 'tsne3'])
    tsne_result_df = pd.concat([entity_df, tsne_df], axis=1)

    # Create a colormap to map each unique pair to a unique color
    pairs = similar_entities_df[['entity1', 'entity2']].values.tolist()
    unique_pairs = list(set(tuple(sorted(pair)) for pair in pairs))
    num_unique_pairs = len(unique_pairs)
    color_map = ListedColormap(plt.cm.tab10(np.linspace(0, 1, num_unique_pairs)))

    # Plot the t-SNE visualization for each pair with unique colors in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    legend_patches = []  # List to store legend patches
    
    for index, (entity1, entity2) in enumerate(unique_pairs):
        color = color_map(index)

        x1, y1, z1 = tsne_result_df.loc[tsne_result_df['entity'] == entity1, ['tsne1', 'tsne2', 'tsne3']].values[0]
        x2, y2, z2 = tsne_result_df.loc[tsne_result_df['entity'] == entity2, ['tsne1', 'tsne2', 'tsne3']].values[0]

        ax.scatter(x1, y1, z1, color=color)
        ax.scatter(x2, y2, z2, color=color)

        # Draw a line connecting the entities
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color)

        # Create a legend patch for each pair
        legend_patches.append(Patch(color=color, label=f'{entity1}-{entity2}'))

    # Add legend
    # ax.legend(handles=legend_patches, loc='upper right')

    ax.set_title('t-SNE Visualization for Each Pair with Unique Colors (3D)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'tsne_3d_unique.png')
    plt.savefig(output_path)
    
    plt.show()

# Example usage:
plot_tsne_2d('D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entities.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entity_embeddings.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/new_file.csv',
             'D:/PE/EduEmbedd/code base/code_2023_fall/Visualization/result')

# plot_tsne_3d('D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entities.csv',
#              'D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/entity_embeddings.csv',
#              'D:/PE/EduEmbedd/code base/code_2023_fall/embeddings_final/transH/transH_50_5_40_0.1_0.1/similar_entities_cos_sim.csv',
#              'D:/PE/EduEmbedd/code base/code_2023_fall/Visualization/result')



