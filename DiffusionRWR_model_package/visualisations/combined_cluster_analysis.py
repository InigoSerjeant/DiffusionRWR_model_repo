"""
Histone Cluster Analysis on Combined Lasso Single-Layer Graph

This script runs cluster analysis on the combined RNA + histone marks 
(K20me3, K9me2) Lasso-regressed single-layer graph.
"""

import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import networkx as nx
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ..graph_generation.generate_graph_internal import lasso_single_graph
from ..run_RWR.numba_RWR import simulate_walks_to_rna_FAST

# Configuration
DATA_FOLDER = str(Path(__file__).resolve().parents[1] / "data" / "Modelled")
LASSO_ALPHA = 0.01
N_CLUSTERS = 5  # Number of clusters for spectral clustering
N_SIMULATIONS = 10000  # Number of random walk simulations
RESTART_PROB = 0.05  # Probability of restart during RWR
START_NODE = 'e3'  # Start from the dominant temporal anchor
N_TOP_GENES = 100  # Number of top genes to report


def _discover_sources() -> dict[str, tuple[Path, str]]:
    data_path = Path(DATA_FOLDER)
    pattern = re.compile(r"^overlap_filtered_(.+)_m_v2\.csv$", re.IGNORECASE)
    source_map: dict[str, tuple[Path, str]] = {}

    for file_path in sorted(data_path.glob("overlap_filtered_*_m_v2.csv")):
        match = pattern.match(file_path.name)
        if not match:
            continue

        token = match.group(1)
        if token.lower().startswith("rna"):
            label = "RNA"
            suffix = "RNA"
        else:
            suffix = token[:1].upper() + token[1:]
            label = suffix

        if label in source_map:
            source_map[f"{label} ({file_path.stem})"] = (file_path, suffix)
        else:
            source_map[label] = (file_path, suffix)

    return source_map


def _node_type(node_name: str) -> str:
    if node_name.startswith('e'):
        return 'basis'
    upper = node_name.upper()
    if upper.endswith('_RNA'):
        return 'RNA'
    suffix = upper.rsplit('_', 1)[-1] if '_' in upper else ''
    if suffix.startswith('K') and 'ME' in suffix:
        return suffix
    return 'other'

def run_combined_cluster_analysis():
    """
    Run histone cluster analysis on combined Lasso single-layer graph.
    """
    print("=" * 80)
    print("COMBINED LASSO SINGLE-LAYER CLUSTER ANALYSIS")
    print("=" * 80)
    
    # Step 1: Load and combine data
    print("\n[STEP 1] Loading combined data...")
    sources = _discover_sources()
    if 'RNA' not in sources:
        raise ValueError("No RNA overlap dataset found (expected overlap_filtered_rna*_m_v2.csv)")

    source_frames = []
    for label, (file_path, suffix) in sources.items():
        frame = pd.read_csv(file_path, index_col=0)
        print(f"  {label}: {frame.shape}")
        frame.index = [f"{g}_{suffix}" for g in frame.index]
        source_frames.append(frame)

    combined_df = pd.concat(source_frames)
    print(f"  Combined shape before filtering: {combined_df.shape}")
    
    # Filter to only integer time points (0, 1, 2, 3, 4)
    integer_cols = [col for col in combined_df.columns if float(col) % 1 == 0]
    combined_df = combined_df[integer_cols]
    print(f"  Combined shape after filtering: {combined_df.shape}")
    
    print(f"  Combined shape after filtering: {combined_df.shape}")
    print(f"  Genes: {combined_df.shape[0]}")
    print(f"  Time points: {combined_df.shape[1]}")
    
    # Step 2: Standardize data (row-wise)
    print("\n[STEP 2] Standardizing data...")
    combined_std = combined_df.sub(combined_df.mean(axis=1), axis=0).div(combined_df.std(axis=1), axis=0)
    print(f"  Data standardized (row-wise z-score)")
    
    # Step 3: Generate Lasso single-layer graph
    print("\n[STEP 3] Generating Lasso single-layer graph...")
    data_dict = {'combined': combined_std}
    import matplotlib
    matplotlib.use('Agg')
    
    adjacency_dict = lasso_single_graph(
        std_data_dict=data_dict,
        edge_fn=None,
        start='e3',
        end='e5',
        return_signs=False
    )
    adjacency = adjacency_dict['combined']
    
    # Threshold small values
    adjacency[adjacency < 0.001] = 0
    
    print(f"  Adjacency matrix shape: {adjacency.shape}")
    print(f"  Number of edges: {(adjacency > 0).sum().sum()}")
    print(f"  Nodes: {adjacency.shape[0]}")
    
    # Get stats about the graph
    edge_counts = (adjacency > 0).sum(axis=1)
    print(f"  Max outgoing edges: {edge_counts.max()} ({adjacency.index[edge_counts.argmax()]})")
    print(f"  Min outgoing edges: {edge_counts.min()}")
    print(f"  Mean outgoing edges: {edge_counts.mean():.2f}")
    
    # Step 4: Identify temporal anchors (basis vectors)
    print("\n[STEP 4] Identifying temporal anchors...")
    basis_indices = [i for i, node in enumerate(adjacency.index) if node.startswith('e')]
    basis_names = [adjacency.index[i] for i in basis_indices]
    print(f"  Found basis vectors: {basis_names}")
    
    for basis in basis_names:
        out_edges = (adjacency.loc[basis] > 0).sum()
        in_edges = (adjacency[basis] > 0).sum()
        print(f"    {basis}: {out_edges} outgoing, {in_edges} incoming edges")
    
    # Step 5: Run RWR from dominant temporal anchor
    print(f"\n[STEP 5] Running random walks from {START_NODE}...")
    print(f"  Simulations: {N_SIMULATIONS}")
    print(f"  Restart probability: {RESTART_PROB}")
    
    try:
        results, n_success, n_restart, n_total, success_list = simulate_walks_to_rna_FAST(
            adj_matrix=adjacency,
            start=START_NODE,
            restart_prob=RESTART_PROB,
            n_sims=N_SIMULATIONS
        )
        print(f"  Successful walks: {n_success}")
        print(f"  Restart events: {n_restart}")
        print(f"  Total steps: {n_total}")
    except Exception as e:
        print(f"  ⚠ Warning: RWR failed with error: {e}")
        print(f"  Skipping RWR analysis, will focus on spectral clustering")
        success_list = None
        n_success = 0
    
    # Step 6: Analyze walk endpoints
    if success_list and len(success_list) > 0:
        print(f"\n[STEP 6] Analyzing walk endpoints...")
        
        endpoint_counts = {}
        for result in success_list:
            end_node = result['end_node']
            endpoint_counts[end_node] = endpoint_counts.get(end_node, 0) + 1
        
        sorted_endpoints = sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {N_TOP_GENES} nodes reached from {START_NODE}:")
        print("=" * 80)
        for i, (node, count) in enumerate(sorted_endpoints[:N_TOP_GENES], 1):
            percentage = 100 * count / n_success
            print(f"{i:3d}. {node:40s}: {count:6d} walks ({percentage:5.1f}%)")
        
        print(f"\nTotal unique nodes reached: {len(endpoint_counts)}")
    
    # Step 7: Spectral clustering on adjacency matrix
    print(f"\n[STEP 7] Running spectral clustering ({N_CLUSTERS} clusters)...")
    
    # Convert to dense if necessary
    adjacency_dense = adjacency.values if hasattr(adjacency, 'values') else adjacency
    
    # Create symmetric adjacency for clustering (average of A and A^T)
    adjacency_symmetric = (adjacency_dense + adjacency_dense.T) / 2
    
    try:
        spectral = SpectralClustering(
            n_clusters=N_CLUSTERS,
            affinity='precomputed',
            random_state=42,
            n_init=10
        )
        cluster_labels = spectral.fit_predict(adjacency_symmetric)
        
        print(f"  Clustering complete")
        
        # Analyze cluster composition
        unique_labels = np.unique(cluster_labels)
        print(f"\nCluster composition:")
        print("=" * 80)
        
        cluster_sizes = {}
        cluster_members = {label: [] for label in unique_labels}
        
        for i, (node, label) in enumerate(zip(adjacency.index, cluster_labels)):
            cluster_members[label].append(node)
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        for label in sorted(unique_labels):
            size = cluster_sizes[label]
            percentage = 100 * size / len(cluster_labels)
            
            # Get composition by type
            members = cluster_members[label]
            type_counts = {}
            for member in members:
                member_type = _node_type(member)
                if member_type == 'basis':
                    continue
                type_counts[member_type] = type_counts.get(member_type, 0) + 1
            basis_count = sum(1 for m in members if m.startswith('e'))
            
            print(f"\nCluster {label}: {size} nodes ({percentage:.1f}%)")
            composition = ", ".join([f"{count} {name}" for name, count in sorted(type_counts.items())])
            print(f"  Composition: {composition}, {basis_count} basis vectors")
            print(f"  Sample members: {', '.join(members[:5])}")
        
        # Export cluster assignments
        cluster_df = pd.DataFrame({
            'node': adjacency.index,
            'cluster': cluster_labels
        })
        
        output_file = "combined_cluster_assignments.csv"
        cluster_df.to_csv(output_file, index=False)
        print(f"\n  Exported cluster assignments to: {output_file}")
        
    except Exception as e:
        print(f"  ⚠ Error during spectral clustering: {e}")
        cluster_labels = None
    
    # Step 8: PCA visualization (if RWR was successful)
    print(f"\n[STEP 8] Creating PCA visualization...")
    
    try:
        # Perform PCA on combined data
        pca = PCA(n_components=3)
        pca.fit(combined_std)
        
        print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}")
        
        # Transform data
        embeddings = pca.transform(combined_std)
        
        # Create interactive 3D scatter plot
        node_types = []
        colors = []
        hover_texts = []
        
        unique_histone_types = sorted({
            _node_type(node) for node in adjacency.index
            if _node_type(node) not in {'RNA', 'basis', 'other'}
        })
        palette = ['red', 'green', 'purple', 'orange', 'brown', 'teal', 'magenta']
        color_map = {'RNA': 'blue', 'basis': 'gold', 'other': 'gray'}
        for i, h_type in enumerate(unique_histone_types):
            color_map[h_type] = palette[i % len(palette)]
        
        for i, node in enumerate(adjacency.index):
            hover_texts.append(f"{node}<br>Cluster: {cluster_labels[i] if cluster_labels is not None else 'N/A'}")
            
            if node.startswith('e'):
                node_types.append('Basis Vector')
                colors.append(color_map['basis'])
            elif '_RNA' in node:
                node_types.append('RNA')
                colors.append(color_map['RNA'])
            else:
                histone_type = _node_type(node)
                node_types.append(histone_type)
                colors.append(color_map.get(histone_type, color_map['other']))
                
        
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                opacity=0.6,
                line=dict(width=0)
            ),
            text=hover_texts,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title="Combined Lasso Network - PCA Visualization",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
            ),
            width=1000,
            height=800
        )
        
        output_file = "combined_pca_clusters.html"
        fig.write_html(output_file)
        print(f"  Visualization saved to: {output_file}")
        
    except Exception as e:
        print(f"  ⚠ PCA visualization failed: {e}")
    
    # Step 9: Network analysis statistics
    print(f"\n[STEP 9] Network statistics...")
    
    # Create NetworkX graph for analysis
    positive_values = adjacency.to_numpy()
    positive_values = positive_values[positive_values > 0]
    if positive_values.size == 0:
        print("  No positive edges found after thresholding")
        return

    threshold_value = np.percentile(positive_values, 90)
    adjacency_threshold = adjacency.gt(threshold_value)
    G = nx.DiGraph()
    
    for i, source in enumerate(adjacency.index):
        for j, target in enumerate(adjacency.index):
            if adjacency_threshold.iloc[i, j]:
                G.add_edge(source, target, weight=adjacency.iloc[i, j])
    
    print(f"  Filtered graph (top 10% edges): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Calculate centrality measures
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    top_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nTop 10 nodes by in-degree (targets of regulation):")
    for node, degree in top_in:
        print(f"  {node:40s}: {degree:4d}")
    
    print(f"\nTop 10 nodes by out-degree (regulators):")
    for node, degree in top_out:
        print(f"  {node:40s}: {degree:4d}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_combined_cluster_analysis()
