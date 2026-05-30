import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..graph_generation.generate_graph_internal import lasso_single_graph

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'Modelled'


def _discover_sources() -> dict[str, tuple[Path, str]]:
    pattern = re.compile(r'^overlap_filtered_(.+)_m_v2\.csv$', re.IGNORECASE)
    source_map: dict[str, tuple[Path, str]] = {}

    for file_path in sorted(DATA_DIR.glob('overlap_filtered_*_m_v2.csv')):
        match = pattern.match(file_path.name)
        if not match:
            continue

        token = match.group(1)
        if token.lower().startswith('rna'):
            label = 'RNA'
            suffix = 'RNA'
        else:
            suffix = token[:1].upper() + token[1:]
            label = suffix

        if label in source_map:
            source_map[f"{label} ({file_path.stem})"] = (file_path, suffix)
        else:
            source_map[label] = (file_path, suffix)

    return source_map


def _node_type(node: str) -> str:
    if node.startswith('e') and len(node) == 2:
        return 'basis'
    upper = node.upper()
    if upper.endswith('_RNA'):
        return 'RNA'
    suffix = upper.rsplit('_', 1)[-1] if '_' in upper else ''
    if suffix.startswith('K') and 'ME' in suffix:
        return suffix
    return 'other'


sources = _discover_sources()
if 'RNA' not in sources:
    raise ValueError('No RNA overlap dataset found (expected overlap_filtered_rna*_m_v2.csv).')

frames = []
for label, (file_path, suffix) in sources.items():
    df = pd.read_csv(file_path, index_col=0)
    print(f"{label} shape: {df.shape}")
    df.index = [f"{g}_{suffix}" for g in df.index]
    frames.append(df)

combined = pd.concat(frames, axis=0)
print(f"\nCombined shape before filtering: {combined.shape}")

integer_cols = [col for col in combined.columns if float(col) % 1 == 0]
combined = combined[integer_cols]
print(f"Combined shape after filtering to integer time points: {combined.shape}")

combined_std = combined.sub(combined.mean(axis=1), axis=0).div(combined.std(axis=1), axis=0)

data_dict = {'combined_rna_histones': combined_std}

print("\nGenerating Lasso graph...")
adjacency_dict = lasso_single_graph(data_dict, edge_fn=None, start='e3', end='e5', return_signs=False)
adjacency_df = adjacency_dict['combined_rna_histones'].copy()
adjacency_df[adjacency_df < 0.001] = 0

graph = nx.from_pandas_adjacency(adjacency_df, create_using=nx.DiGraph)

print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(20, 20))
pos = nx.spring_layout(graph, seed=42, iterations=50, k=0.5)

basis_vectors = ['e1', 'e2', 'e3', 'e4', 'e5']
basis_time = {'e1': 'T1', 'e2': 'T2', 'e3': 'T3', 'e4': 'T4', 'e5': 'T5'}
basis_color_map = {'e1': 'gold', 'e2': 'orange', 'e3': 'red', 'e4': 'purple', 'e5': 'darkblue'}
closest_to_basis = {}

for basis in basis_vectors:
    if basis in pos:
        basis_pos = pos[basis]
        distances = {}
        for node in graph.nodes():
            if node == basis:
                continue
            node_pos = pos[node]
            dist = np.sqrt((node_pos[0] - basis_pos[0]) ** 2 + (node_pos[1] - basis_pos[1]) ** 2)
            distances[node] = dist
        if distances:
            closest_node = min(distances, key=distances.get)
            closest_to_basis[basis] = closest_node
            print(f"Closest to {basis}: {closest_node} (distance: {distances[closest_node]:.4f})")

palette = ['lightcoral', 'lightgreen', 'lightsalmon', 'plum', 'khaki', 'lightpink']
all_types = sorted({_node_type(node) for node in graph.nodes() if _node_type(node) not in {'basis', 'RNA', 'other'}})
type_color_map = {'RNA': 'lightblue', 'other': 'gray'}
for i, t in enumerate(all_types):
    type_color_map[t] = palette[i % len(palette)]

node_colors = []
node_sizes = []
for node in graph.nodes():
    anchored = False
    for basis, closest_node in closest_to_basis.items():
        if node == closest_node:
            node_colors.append(basis_color_map[basis])
            node_sizes.append(500)
            anchored = True
            break

    if anchored:
        continue

    node_t = _node_type(node)
    node_colors.append(type_color_map.get(node_t, 'gray'))
    node_sizes.append(100)

nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.8)
nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3, ax=ax, arrowsize=5, width=0.5)

for basis, closest_node in closest_to_basis.items():
    x, y = pos[closest_node]
    ax.text(
        x,
        y,
        basis_time[basis],
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=basis_color_map[basis], alpha=0.7),
        ha='center',
        va='center',
    )

ax.set_title(
    f'Combined RNA + Histone Modifications Network (Temporal Structure)\n'
    f'Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}\n'
    f'Large nodes: Temporal anchors closest to each basis vector',
    fontsize=14,
)
ax.axis('off')

legend_elements = [
    Patch(facecolor=type_color_map['RNA'], label=f"RNA ({sum(1 for n in graph.nodes() if _node_type(n) == 'RNA')} nodes)"),
]
for t in all_types:
    legend_elements.append(Patch(facecolor=type_color_map[t], label=f"{t} ({sum(1 for n in graph.nodes() if _node_type(n) == t)} nodes)"))
legend_elements.extend([
    Patch(facecolor='gold', label='Temporal anchor (T1/e1)'),
    Patch(facecolor='orange', label='Temporal anchor (T2/e2)'),
    Patch(facecolor='red', label='Temporal anchor (T3/e3)'),
    Patch(facecolor='purple', label='Temporal anchor (T4/e4)'),
    Patch(facecolor='darkblue', label='Temporal anchor (T5/e5)'),
])
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

fig.savefig('lasso_combined_rna_histones.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved: lasso_combined_rna_histones.png")
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
