import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .preprocess_data import load_data
from .generate_graph_internal import generate_single_layer_graphs
from .generate_multi_graph import create_multigraph
from .edge_weight_functions import corr_power, cor_gaussian_shifted


def visualize_single_layer_graph(folder_cat='rna', folder_name="Modelled", edge_fn=corr_power, 
                        start='e3', end='e5', threshold=0.1, 
                        top_n_genes=50, figsize=(15, 12)):
    """
    Load RNA data, create a single-layer graph, and visualize it.
    
    Parameters:
    -----------
    folder_cat : str
        Name of the folder within the package's data directory (default: "Modelled")
    edge_fn : function
        Edge weight function to use (default: corr_power)
    start : str
        Start basis vector (default: 'e3')
    end : str
        End basis vector (default: 'e5')
    threshold : float
        Edge weight threshold for visualization (default: 0.1)
    top_n_genes : int
        Number of top genes to include in visualization based on degree (default: 50)
    figsize : tuple
        Figure size for the plot (default: (15, 12))
    
    Returns:
    --------
    tuple
        (data_dict, adjacency_dict, fig, ax) - data, adjacency matrices, figure, and axes
    """
    # Step 1: Load and preprocess data
    print("=" * 60)
    print("STEP 1: Loading and preprocessing RNA data")
    print("=" * 60)
    data_dict = load_data(folder_name)
    
    # Step 2: Generate single-layer graph
    print("\n" + "=" * 60)
    print("STEP 2: Generating single-layer graph")
    print("=" * 60)
    adjacency_dict = generate_single_layer_graphs(
        data_dict, 
        edge_fn=edge_fn, 
        start=start, 
        end=end
    )
    
    # Step 3: Visualize the graph
    print("\n" + "=" * 60)
    print("STEP 3: Visualizing the graph")
    print("=" * 60)
    
    # Get the first dataset's adjacency matrix for visualization
    dataset_name = list([i for i in adjacency_dict.keys() if folder_cat in i])[0]
    adjacency_df = adjacency_dict[dataset_name]
    
    print(f"\nVisualizing dataset: {dataset_name}")
    
    # Filter edges by threshold
    adjacency_filtered = adjacency_df.copy()
    adjacency_filtered[adjacency_filtered < threshold] = 0
    
    # Calculate node degrees to find most connected genes
    degrees = (adjacency_filtered > 0).sum(axis=1)
    top_genes = degrees.nlargest(top_n_genes).index.tolist()
    
    # Include start and end nodes
    for node in [start, end]:
        if node not in top_genes and node in adjacency_filtered.index:
            top_genes.append(node)
    
    # Create subgraph with top genes
    adjacency_subgraph = adjacency_filtered.loc[top_genes, top_genes]
    
    print(f"Filtered edges with threshold > {threshold}")
    print(f"Showing top {top_n_genes} genes by degree")
    print(f"Subgraph has {len(top_genes)} nodes")
    print(f"Number of edges: {(adjacency_subgraph > 0).sum().sum() // 2}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot adjacency matrix as heatmap
    im = ax.imshow(adjacency_subgraph.values, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(top_genes)))
    ax.set_yticks(range(len(top_genes)))
    ax.set_xticklabels(top_genes, rotation=90, fontsize=6)
    ax.set_yticklabels(top_genes, fontsize=6)
    
    # Highlight start and end nodes
    start_idx = top_genes.index(start) if start in top_genes else None
    end_idx = top_genes.index(end) if end in top_genes else None
    
    if start_idx is not None:
        ax.axhline(y=start_idx, color='lightgreen', linewidth=2, alpha=0.7)
        ax.axvline(x=start_idx, color='lightgreen', linewidth=2, alpha=0.7)
    
    if end_idx is not None:
        ax.axhline(y=end_idx, color='lightcoral', linewidth=2, alpha=0.7)
        ax.axvline(x=end_idx, color='lightcoral', linewidth=2, alpha=0.7)
    
    # Title and formatting
    ax.set_title(f"Adjacency Matrix - {dataset_name}\n"
                f"Edge function: {edge_fn.__name__}, Threshold: {threshold}\n"
                f"Start: {start}, End: {end}",
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Gene Index', fontsize=10)
    ax.set_ylabel('Gene Index', fontsize=10)
    
    plt.tight_layout()
    
    print("\nVisualization complete!")
    print(f"Green lines: {start} (start)")
    print(f"Red lines: {end} (end)")
    
    return data_dict, adjacency_dict, fig, ax


def save_graph_visualization(folder_name="Modelled", edge_fn=corr_power,
                             start='e3', end='e5', threshold=0.1,
                             top_n_genes=50, output_path="rna_graph.png"):
    """
    Create and save RNA graph visualization to file.
    
    Parameters:
    -----------
    folder_name : str
        Name of the folder within the package's data directory
    edge_fn : function
        Edge weight function to use
    start : str
        Start basis vector
    end : str
        End basis vector
    threshold : float
        Edge weight threshold for visualization
    top_n_genes : int
        Number of top genes to include in visualization
    output_path : str
        Path to save the output image
    """
    data_dict, adjacency_dict, fig, ax = visualize_single_layer_graph(
        folder_cat='rna',
        folder_name=folder_name,
        edge_fn=edge_fn,
        start=start,
        end=end,
        threshold=threshold,
        top_n_genes=top_n_genes
    )
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {output_path}")
    plt.close(fig)
    
    return data_dict, adjacency_dict


def visualize_multi_layer_graph(folder_name="Modelled", edge_fn_intra=corr_power, 
                                edge_fn_inter=lambda x, y: np.corrcoef(x, y)[0, 1],
                                start='e3', end='e5', threshold=0.1, 
                                top_n_genes_per_layer=30, figsize=(20, 18)):
    """
    Load data, create a multi-layer graph, and visualize it as adjacency matrix.
    
    Parameters:
    -----------
    folder_name : str
        Name of the folder within the package's data directory (default: "Modelled")
    edge_fn_intra : function
        Edge weight function for intra-layer connections (default: corr_power)
    edge_fn_inter : function
        Edge weight function for inter-layer connections (default: correlation)
    start : str
        Start basis vector (default: 'e3')
    end : str
        End basis vector (default: 'e5')
    threshold : float
        Edge weight threshold for filtering (default: 0.1)
    top_n_genes_per_layer : int
        Number of top genes per layer to include (default: 30)
    figsize : tuple
        Figure size for the plot (default: (20, 18))
    
    Returns:
    --------
    tuple
        (data_dict, intra_layer_graphs, complete_adj, fig, ax)
    """
    # Step 1: Load and preprocess data
    print("=" * 60)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 60)
    data_dict = load_data(folder_name)
    
    # Step 2: Generate intra-layer graphs
    print("\n" + "=" * 60)
    print("STEP 2: Generating intra-layer graphs")
    print("=" * 60)
    intra_layer_graphs = generate_single_layer_graphs(
        data_dict, 
        edge_fn=edge_fn_intra, 
        start=start, 
        end=end
    )
    
    # Step 3: Create multi-layer graph
    print("\n" + "=" * 60)
    print("STEP 3: Creating multi-layer graph")
    print("=" * 60)
    complete_adj = create_multigraph(
        data_dict,
        intra_layer_graphs,
        edge_fn_inter=edge_fn_inter,
        start=start,
        end=end
    )
    
    # Step 4: Filter and visualize
    print("\n" + "=" * 60)
    print("STEP 4: Visualizing multi-layer graph")
    print("=" * 60)
    
    # Filter adjacency matrix by threshold
    complete_adj_filtered = complete_adj.copy()
    complete_adj_filtered[complete_adj_filtered < threshold] = 0
    
    # Select top genes per layer based on degree
    selected_nodes = []
    layer_labels = []
    layer_boundaries = [0]
    
    for layer_name, intra_adj in intra_layer_graphs.items():
        # Get nodes for this layer in the complete graph
        layer_nodes = [node for node in complete_adj.index if node.endswith(f'_{layer_name}')]
        
        # Calculate degrees for this layer
        layer_subgraph = complete_adj_filtered.loc[layer_nodes, layer_nodes]
        degrees = (layer_subgraph > 0).sum(axis=1)
        
        # Get top N genes for this layer
        top_genes = degrees.nlargest(top_n_genes_per_layer).index.tolist()
        
        # Always include start and end nodes if they exist
        for node_base in [start, end]:
            node_full = f'{node_base}_{layer_name}'
            if node_full in layer_nodes and node_full not in top_genes:
                top_genes.append(node_full)
        
        selected_nodes.extend(top_genes)
        layer_labels.extend([layer_name] * len(top_genes))
        layer_boundaries.append(layer_boundaries[-1] + len(top_genes))
        
        print(f"\nLayer {layer_name}: selected {len(top_genes)} nodes")
    
    # Create subgraph with selected nodes
    adjacency_subgraph = complete_adj_filtered.loc[selected_nodes, selected_nodes]
    
    print(f"\nTotal nodes in visualization: {len(selected_nodes)}")
    print(f"Number of edges: {(adjacency_subgraph > 0).sum().sum() // 2}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot adjacency matrix as heatmap
    im = ax.imshow(adjacency_subgraph.values, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    
    # Add grid lines to separate layers
    for boundary in layer_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color='red', linewidth=2, alpha=0.5)
        ax.axvline(x=boundary - 0.5, color='red', linewidth=2, alpha=0.5)
    
    # Set ticks and labels (show only every Nth label to avoid clutter)
    label_step = max(1, len(selected_nodes) // 50)
    tick_positions = list(range(0, len(selected_nodes), label_step))
    tick_labels = [selected_nodes[i].split('_')[0] if i < len(selected_nodes) else '' 
                   for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)
    
    # Add layer annotations
    unique_layers = list(dict.fromkeys(layer_labels))  # Preserve order
    for i, layer in enumerate(unique_layers):
        # Find the center position of this layer
        layer_indices = [j for j, l in enumerate(layer_labels) if l == layer]
        center = (layer_indices[0] + layer_indices[-1]) / 2
        
        # Add text annotation
        ax.text(-0.02, center, layer, transform=ax.get_yaxis_transform(),
               ha='right', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(center, -0.02, layer, transform=ax.get_xaxis_transform(),
               ha='center', va='top', fontsize=10, fontweight='bold', rotation=0,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Title and formatting
    ax.set_title(f"Multi-Layer Graph Adjacency Matrix\n"
                f"Intra-layer: {edge_fn_intra.__name__}, Inter-layer: {edge_fn_inter.__name__}\n"
                f"Threshold: {threshold}, Start: {start}, End: {end}",
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Node Index (by layer)', fontsize=12, labelpad=10)
    ax.set_ylabel('Node Index (by layer)', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    
    print("\nVisualization complete!")
    print(f"Red lines: layer boundaries")
    print(f"Layers: {unique_layers}")
    
    return data_dict, intra_layer_graphs, complete_adj, fig, ax


def save_multi_layer_visualization(folder_name="Modelled", edge_fn_intra=corr_power,
                                   edge_fn_inter=lambda x, y: np.corrcoef(x, y)[0, 1],
                                   start='e3', end='e5', threshold=0.1,
                                   top_n_genes_per_layer=30, output_path="multi_layer_graph.png"):
    """
    Create and save multi-layer graph visualization to file.
    
    Parameters:
    -----------
    folder_name : str
        Name of the folder within the package's data directory
    edge_fn_intra : function
        Edge weight function for intra-layer connections
    edge_fn_inter : function
        Edge weight function for inter-layer connections
    start : str
        Start basis vector
    end : str
        End basis vector
    threshold : float
        Edge weight threshold for visualization
    top_n_genes_per_layer : int
        Number of top genes per layer to include
    output_path : str
        Path to save the output image
    """
    data_dict, intra_layer_graphs, complete_adj, fig, ax = visualize_multi_layer_graph(
        folder_name=folder_name,
        edge_fn_intra=edge_fn_intra,
        edge_fn_inter=edge_fn_inter,
        start=start,
        end=end,
        threshold=threshold,
        top_n_genes_per_layer=top_n_genes_per_layer
    )
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {output_path}")
    plt.close(fig)
    
    return data_dict, intra_layer_graphs, complete_adj


# Run when script is executed directly
if __name__ == "__main__":
    # Example usage - uncomment the one you want to run
    
    # Single layer visualization
    # data_dict, adjacency_dict, fig, ax = visualize_single_layer_graph(
    #     folder_cat='rna',
    #     folder_name="Modelled",
    #     edge_fn=corr_power,
    #     start='e3',
    #     end='e5',
    #     threshold=0.1,
    #     top_n_genes=50
    # )
    
    # Multi-layer visualization
    data_dict, intra_graphs, complete_adj, fig, ax = visualize_multi_layer_graph(
        folder_name="Modelled",
        edge_fn_intra=corr_power,
        start='e3',
        end='e5',
        threshold=0.1,
        top_n_genes_per_layer=30
    )
    plt.show()
