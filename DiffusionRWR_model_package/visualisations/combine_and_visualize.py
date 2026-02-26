import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load datasets
data_dir = r'C:\Users\inigo\OneDrive\Documents\Fourth_Year\Computations\Dissertation\DiffusionRWR_model_repo\DiffusionRWR_model_package\data\Modelled'

# Load each dataset
rna = pd.read_csv(os.path.join(data_dir, 'overlap_filtered_rna_ai_m_v2.csv'), index_col=0)
k20me3 = pd.read_csv(os.path.join(data_dir, 'overlap_filtered_k20me3_m_v2.csv'), index_col=0)
k9me2 = pd.read_csv(os.path.join(data_dir, 'overlap_filtered_k9me2_m_v2.csv'), index_col=0)

print(f"RNA shape: {rna.shape}")
print(f"K20me3 shape: {k20me3.shape}")
print(f"K9me2 shape: {k9me2.shape}")

# Rename indices to indicate data type
rna.index = [f"{g}_RNA" for g in rna.index]
k20me3.index = [f"{g}_K20me3" for g in k20me3.index]
k9me2.index = [f"{g}_K9me2" for g in k9me2.index]

# Combine datasets
combined = pd.concat([rna, k20me3, k9me2], axis=0)
print(f"\nCombined shape before filtering: {combined.shape}")
print(f"Columns: {combined.columns.tolist()}")

# Filter to only integer time points (0, 1, 2, 3, 4)
integer_cols = [col for col in combined.columns if float(col) % 1 == 0]
combined = combined[integer_cols]
print(f"Combined shape after filtering to integer time points: {combined.shape}")

# Standardize row-wise
combined_std = combined.sub(combined.mean(axis=1), axis=0).div(combined.std(axis=1), axis=0)

# Load Lasso graph generation
from ..graph_generation.generate_graph_internal import lasso_single_graph
import matplotlib
matplotlib.use('Agg')

data_dict = {'combined_rna_histones': combined_std}

print("\nGenerating Lasso graph...")
adjacency_dict = lasso_single_graph(data_dict, edge_fn=None, start='e3', end='e5', return_signs=False)

adjacency_df = adjacency_dict['combined_rna_histones'].copy()
adjacency_df[adjacency_df < 0.001] = 0

graph = nx.from_pandas_adjacency(adjacency_df, create_using=nx.DiGraph)

print(f"\nCreating visualization...")
fig, ax = plt.subplots(figsize=(20, 20))
pos = nx.spring_layout(graph, seed=42, iterations=50, k=0.5)

# Find nodes closest to each basis vector
basis_vectors = ['e1', 'e2', 'e3', 'e4', 'e5']
basis_time = {'e1': 'T1', 'e2': 'T2', 'e3': 'T3', 'e4': 'T4', 'e5': 'T5'}
basis_color_map = {'e1': 'gold', 'e2': 'orange', 'e3': 'red', 'e4': 'purple', 'e5': 'darkblue'}
closest_to_basis = {}
basis_positions = {}

for basis in basis_vectors:
    if basis in pos:
        basis_pos = pos[basis]
        basis_positions[basis] = basis_pos
        # Calculate distances to all other nodes
        distances = {}
        for node in graph.nodes():
            if node != basis:
                node_pos = pos[node]
                dist = np.sqrt((node_pos[0] - basis_pos[0])**2 + (node_pos[1] - basis_pos[1])**2)
                distances[node] = dist
        # Find closest node
        closest_node = min(distances, key=distances.get)
        closest_to_basis[basis] = closest_node
        print(f"Closest to {basis}: {closest_node} (distance: {distances[closest_node]:.4f})")

# Color code nodes
node_colors = []
node_sizes = []

for node in graph.nodes():
    # Check if this node is closest to a basis vector
    is_closest = False
    for basis, closest_node in closest_to_basis.items():
        if node == closest_node:
            node_colors.append(basis_color_map[basis])
            node_sizes.append(500)  # Larger size for temporal anchor nodes
            is_closest = True
            break
    
    if not is_closest:
        # Color by data type
        if 'RNA' in node:
            node_colors.append('lightblue')
        elif 'K20me3' in node:
            node_colors.append('lightcoral')
        else:  # K9me2
            node_colors.append('lightgreen')
        node_sizes.append(100)

nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.8)
nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3, ax=ax, arrowsize=5, width=0.5)

# Label the temporal anchor nodes
for basis, closest_node in closest_to_basis.items():
    x, y = pos[closest_node]
    ax.text(x, y, basis_time[basis], fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor=basis_color_map[basis], alpha=0.7),
            ha='center', va='center')

ax.set_title(f'Combined RNA + Histone Modifications Network (Temporal Structure)\nNodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}\nLarge nodes: Temporal anchors closest to each basis vector', fontsize=14)
ax.axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', label=f'RNA ({sum(1 for n in graph.nodes() if "RNA" in n)} nodes)'),
    Patch(facecolor='lightcoral', label=f'K20me3 ({sum(1 for n in graph.nodes() if "K20me3" in n)} nodes)'),
    Patch(facecolor='lightgreen', label=f'K9me2 ({sum(1 for n in graph.nodes() if "K9me2" in n)} nodes)'),
    Patch(facecolor='gold', label='Temporal anchor (T1/e1)'),
    Patch(facecolor='orange', label='Temporal anchor (T2/e2)'),
    Patch(facecolor='red', label='Temporal anchor (T3/e3)'),
    Patch(facecolor='purple', label='Temporal anchor (T4/e4)'),
    Patch(facecolor='darkblue', label='Temporal anchor (T5/e5)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

fig.savefig('lasso_combined_rna_histones.png', dpi=100, bbox_inches='tight')
print(f"\nVisualization saved: lasso_combined_rna_histones.png")
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
print("\n=== TEMPORAL STRUCTURE (Basis Vector Anchors) ===")
for basis in basis_vectors:
    if basis in closest_to_basis:
        closest_node = closest_to_basis[basis]
        print(f"{basis_time[basis]} ({basis}): {closest_node}")
print("\n=== LEGEND ===")
print("Data types:")
print("  Blue = RNA expression")
print("  Red = K20me3 histone modification")
print("  Green = K9me2 histone modification")
print("\nTemporal anchors (nodes closest to each basis vector):")
print("  Gold (T1) - Early timepoint")
print("  Orange (T2)")
print("  Red (T3) - Middle timepoint")
print("  Purple (T4)")
print("  Dark Blue (T5) - Late timepoint")
