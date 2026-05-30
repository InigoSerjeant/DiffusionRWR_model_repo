import pandas as pd
import numpy as np

def create_connected_sections(data_A, data_B, label_A, label_B, edge_fn, same_gene=True, return_sign_matrix=False): # Reviewed
    A_new_row_names = [i + label_A for i in data_A.index]
    B_new_col_names = [i + label_B for i in data_B.index]
    connected_section_array = np.zeros((len(A_new_row_names), len(B_new_col_names)))
    
    if same_gene:
        for i, A in enumerate(data_A.index):
            for j, B in enumerate(data_B.index):
                if A.split('*')[0] in B:
                    if return_sign_matrix:
                        # For sign matrix, use raw correlation (not transformed)
                        raw_corr = np.corrcoef(data_A.loc[A], data_B.loc[B])[0, 1]
                        connected_section_array[i,j] = np.sign(raw_corr)
                    else:
                        edge_value = edge_fn(data_A.loc[A], data_B.loc[B])
                        connected_section_array[i,j] = edge_value
    else:
        for i, A in enumerate(data_A.index):
            for j, B in enumerate(data_B.index):
                if return_sign_matrix:
                    # For sign matrix, use raw correlation (not transformed)
                    raw_corr = np.corrcoef(data_A.loc[A], data_B.loc[B])[0, 1]
                    connected_section_array[i,j] = np.sign(raw_corr)
                else:
                    edge_value = edge_fn(data_A.loc[A], data_B.loc[B])
                    connected_section_array[i,j] = edge_value
    
    connected_section = pd.DataFrame(connected_section_array, index=A_new_row_names, columns=B_new_col_names)
    return connected_section

def create_multigraph(std_data_dict, intra_layer_graphs, edge_fn_inter, start='e3', end='e5'): #reviewed
    """
    Create a multi-layer graph with:
    - Intra-layer connections within each dataset
    - Inter-layer connections between datasets
    - Merged basis vectors (e.g., e1_rna_ai, e1_k9me2 merged into e1)
    
    Parameters:
    -----------
    std_data_dict : dict
        Dictionary of standardized datasets (e.g., {'rna': rna_std, 'k9me2': k9me2_std})
    intra_layer_graphs : dict
        Dictionary of intra-layer adjacency matrices
    edge_fn_inter : function
        Function to compute edge weights between layers (takes two node vectors)
    start : str
        Starting basis vector (e.g., 'e3')
    end : str
        Ending basis vector (e.g., 'e5')
    
    Returns:
    --------
    pd.DataFrame : Complete multi-graph adjacency matrix
    """
    
    print("=" * 80)
    print("CREATING MULTI-LAYER GRAPH")
    print("=" * 80)
    
    # Step 1: Generate intra-layer graphs for each dataset
    print("\n[STEP 1] Intra layer graphs are generated separately and provided as input (skipping generation here)")

    
    # Step 2: Generate inter-layer connections
    print("\n[STEP 2] Generating inter-layer connections...")
    dataset_names = list(std_data_dict.keys())
    inter_layer_sections = {}
    
    for i, name_A in enumerate(dataset_names):
        for name_B in dataset_names[i+1:]:  # Only upper triangle to avoid duplicates
            print(f"\n  Connecting {name_A} -> {name_B}")
            
            # Create connections from A to B
            section_AB = create_connected_sections(
                std_data_dict[name_A], 
                std_data_dict[name_B],
                label_A=f'_{name_A}',
                label_B=f'_{name_B}',
                edge_fn=edge_fn_inter,
                same_gene=True
            )
            inter_layer_sections[f'{name_A}_to_{name_B}'] = section_AB
            
            # Create connections from B to A (transpose)
            section_BA = section_AB.T
            section_BA.index = [i + f'_{name_B}' for i in std_data_dict[name_B].index]
            section_BA.columns = [i + f'_{name_A}' for i in std_data_dict[name_A].index]
            inter_layer_sections[f'{name_B}_to_{name_A}'] = section_BA
            
            print(f"    Section shape: {section_AB.shape}")
    
    # Step 3: Assemble the complete multi-graph
    print("\n[STEP 3] Assembling complete multi-graph adjacency matrix...")
    
    # Rename indices for intra-layer graphs to include layer labels
    labeled_intra_graphs = {}
    for name, adj_df in intra_layer_graphs.items():
        new_index = [f"{idx}_{name}" for idx in adj_df.index]
        new_columns = [f"{col}_{name}" for col in adj_df.columns]
        labeled_df = adj_df.copy()
        labeled_df.index = new_index
        labeled_df.columns = new_columns
        labeled_intra_graphs[name] = labeled_df
    
    # Get all unique nodes across all layers
    all_nodes = []
    for name in dataset_names:
        all_nodes.extend(labeled_intra_graphs[name].index.tolist())
    
    # Initialize complete adjacency matrix
    complete_adj = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
    
    # Fill in intra-layer connections
    print("\n  Filling intra-layer connections...")
    for name, adj_df in labeled_intra_graphs.items():
        nodes = adj_df.index.tolist()
        complete_adj.loc[nodes, nodes] = adj_df.values
        print(f"    {name}: {len(nodes)} nodes")
    
    # Fill in inter-layer connections
    print("\n  Filling inter-layer connections...")
    for section_name, section_df in inter_layer_sections.items():
        row_nodes = section_df.index.tolist()
        col_nodes = section_df.columns.tolist()
        complete_adj.loc[row_nodes, col_nodes] = section_df.values
        print(f"    {section_name}: {section_df.shape}")

    # Step 4: Merge basis vectors across layers
    print("\n[STEP 4] Merging basis vectors across layers...")
    basis_names = ['e1', 'e2', 'e3', 'e4', 'e5']
    
    for basis_name in basis_names:
        # Find all nodes that are exactly this basis vector with layer suffix (e.g., "e1_rna_ai", "e1_k9me2")
        nodes_to_merge = [node for node in complete_adj.index if node.startswith(basis_name + '_') and any(node.endswith(f'_{layer}') for layer in dataset_names)]
        
        if len(nodes_to_merge) > 1:
            print(f"\n  Merging {basis_name}: {nodes_to_merge}")
            
            # Create merged node with combined edges
            # For incoming edges: sum all edges pointing to any of the nodes_to_merge
            incoming_edges = complete_adj.loc[:, nodes_to_merge].sum(axis=1)
            
            # For outgoing edges: sum all edges from any of the nodes_to_merge
            outgoing_edges = complete_adj.loc[nodes_to_merge, :].sum(axis=0)
            
            # Remove old nodes
            complete_adj = complete_adj.drop(index=nodes_to_merge, columns=nodes_to_merge)
            
            # Add merged node
            # First, add as new row and column with zeros
            complete_adj[basis_name] = 0.0
            complete_adj.loc[basis_name] = 0.0
            
            # Set incoming edges (exclude the basis nodes we're merging)
            other_nodes = [n for n in incoming_edges.index if n not in nodes_to_merge]
            complete_adj.loc[other_nodes, basis_name] = incoming_edges.loc[other_nodes].values
            
            # Set outgoing edges (exclude the basis nodes we're merging)
            other_nodes = [n for n in outgoing_edges.index if n not in nodes_to_merge]
            complete_adj.loc[basis_name, other_nodes] = outgoing_edges.loc[other_nodes].values
            
            print(f"    Created merged node '{basis_name}' with {(complete_adj.loc[basis_name] != 0).sum()} outgoing and {(complete_adj[basis_name] != 0).sum()} incoming edges")

    # Print statistics
    print("\n" + "=" * 80)
    print("MULTI-GRAPH STATISTICS")
    print("=" * 80)
    basis_count = sum(1 for node in complete_adj.index if node in basis_names)
    gene_count = complete_adj.shape[0] - basis_count
    print(f"Total nodes: {complete_adj.shape[0]} ({gene_count} genes + {basis_count} merged basis vectors)")
    print(f"Total possible edges: {complete_adj.shape[0] * complete_adj.shape[1]}")
    print(f"Non-zero edges: {(complete_adj != 0).sum().sum()}")
    print(f"Edge density: {(complete_adj != 0).sum().sum() / (complete_adj.shape[0] * complete_adj.shape[1]):.4f}")
    print(f"\nEdge weight statistics:")
    print(f"  Min: {complete_adj.values.min():.6f}")
    print(f"  Max: {complete_adj.values.max():.6f}")
    print(f"  Mean: {complete_adj.values.mean():.6f}")
    print(f"  Median: {np.median(complete_adj.values):.6f}")
    return complete_adj

def create_multigraph_with_layer_transitions(std_data_dict, intra_layer_graphs, edge_fn_inter, alpha=0.1, start='e3', end='e5'): # Reviewed
    """
    Create a multi-layer graph with controlled inter-layer transitions and normalized edge weights for random walks.
    
    This function first creates a standard multi-layer graph using create_multigraph(), then scales the edge weights
    to control inter-layer transition probability and normalizes for random walk.
    
    At each step in a random walk, there's a probability alpha of transitioning to a different layer,
    and probability (1-alpha) of staying within the current layer. Edge weights are normalized so that
    outgoing edge weights from each node sum to 1, making the adjacency matrix a proper transition matrix.
    
    Parameters:
    -----------
    std_data_dict : dict
        Dictionary of standardized datasets (e.g., {'rna': rna_std, 'k9me2': k9me2_std})
    intra_layer_graphs : dict
        Dictionary of intra-layer adjacency matrices
    edge_fn_inter : function
        Function to compute edge weights between layers (takes two node vectors)
    alpha : float, default=0.1
        Probability of transitioning to a different layer at each step (0 <= alpha <= 1)
        - alpha = 0: no inter-layer transitions (isolated layers)
        - alpha = 1: only inter-layer transitions (no intra-layer movement)
        - alpha = 0.1: 10% chance of layer transition, 90% chance of staying in layer
    start : str
        Starting basis vector (e.g., 'e3')
    end : str
        Ending basis vector (e.g., 'e5')
    
    Returns:
    --------
    pd.DataFrame : Complete multi-graph adjacency matrix with normalized edge weights (transition probabilities)
    """
    
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    print("=" * 80)
    print("CREATING MULTI-LAYER GRAPH WITH LAYER TRANSITIONS")
    print("=" * 80)
    print(f"Alpha (inter-layer transition probability): {alpha}")
    print(f"Intra-layer probability: {1 - alpha}")
    
    # Step 1: Create standard multi-layer graph
    complete_adj = create_multigraph(std_data_dict, intra_layer_graphs, edge_fn_inter, start, end)
    
    # Step 2: Identify layers and scale edges based on alpha
    print("\n[STEP 2] Scaling edge weights based on alpha parameter...")
    dataset_names = list(std_data_dict.keys())
    basis_names = ['e1', 'e2', 'e3', 'e4', 'e5']
    num_layers = len(dataset_names)
    
    # Determine which nodes belong to which layer
    def get_node_layer(node):
        """Determine which layer a node belongs to."""
        if node in basis_names:
            return None  # Basis nodes are shared across layers
        for layer in dataset_names:
            if node.endswith(f'_{layer}'):
                return layer
        return None
    
    # Create a mapping of nodes to layers
    node_layers = {node: get_node_layer(node) for node in complete_adj.index}
    
    # Step 2a: Normalize edges BEFORE applying alpha scaling
    print("\n[STEP 2a] Normalizing edge weights before scaling...")
    
    # For each node, we need to:
    # 1. Separate intra-layer and inter-layer edges
    # 2. Normalize each group to sum to 1
    # 3. Scale: intra-layer by (1-alpha), inter-layer by alpha/(num_layers-1)
    
    scaled_adj = pd.DataFrame(0.0, index=complete_adj.index, columns=complete_adj.columns)
    
    for source_node in complete_adj.index:
        source_layer = node_layers[source_node]
        
        # Get all outgoing edges for this node
        outgoing_edges = complete_adj.loc[source_node, :]
        nonzero_targets = outgoing_edges[outgoing_edges > 0].index.tolist()
        
        if len(nonzero_targets) == 0:
            continue
        
        # Categorize edges
        intra_targets = []
        inter_targets = []
        
        for target in nonzero_targets:
            target_layer = node_layers[target]
            
            # Check if intra-layer or inter-layer
            if source_layer is None or target_layer is None:
                # Basis vector involved - treat as intra for now
                intra_targets.append(target)
            elif source_layer == target_layer:
                # Same layer
                intra_targets.append(target)
            else:
                # Different layer
                inter_targets.append(target)
        
        # Get raw weights
        intra_weights = outgoing_edges[intra_targets].values if len(intra_targets) > 0 else np.array([])
        inter_weights = outgoing_edges[inter_targets].values if len(inter_targets) > 0 else np.array([])
        
        # Normalize within each category
        intra_sum = intra_weights.sum()
        inter_sum = inter_weights.sum()
        
        # Scale by probability factors
        if intra_sum > 0:
            intra_normalized = (intra_weights / intra_sum) * (1 - alpha)
            scaled_adj.loc[source_node, intra_targets] = intra_normalized
        
        if inter_sum > 0:
            inter_normalized = (inter_weights / inter_sum) * alpha
            scaled_adj.loc[source_node, inter_targets] = inter_normalized
    
    # Count edge types for diagnostics
    intra_count = 0
    inter_count = 0
    
    for source_node in scaled_adj.index:
        source_layer = node_layers[source_node]
        for target_node in scaled_adj.columns:
            if scaled_adj.loc[source_node, target_node] == 0:
                continue
            target_layer = node_layers[target_node]
            
            if source_layer is None or target_layer is None or source_layer == target_layer:
                intra_count += 1
            else:
                inter_count += 1
    
    print("  ✓ Edge scaling complete with proper normalization")
    print(f"    Intra-layer edges: {intra_count}")
    print(f"    Inter-layer edges: {inter_count}")
    
    # Verify that row sums equal 1 (within tolerance)
    row_sums = scaled_adj.sum(axis=1)
    non_zero_rows = row_sums[row_sums > 0]
    if len(non_zero_rows) > 0:
        print(f"\n  Row sum verification (should be ~1.0 for nodes with edges):")
        print(f"    Min: {non_zero_rows.min():.6f}")
        print(f"    Max: {non_zero_rows.max():.6f}")
        print(f"    Mean: {non_zero_rows.mean():.6f}")
        if not np.allclose(non_zero_rows, 1.0, rtol=1e-5):
            print(f"    ⚠ WARNING: Row sums deviate from 1.0!")
    
    # Step 3: Check edge weight distributions
    print("\n[STEP 3] Verifying edge weight scaling...")
    print("  Edges are normalized: each node's outgoing edges sum to 1.0")
    print("  Alpha parameter ensures correct probability distribution:")
    print(f"    P(stay in layer) = {1-alpha:.3f}")
    print(f"    P(switch layer) = {alpha:.3f}")
    
    # Check row sums are 1
    row_sums = scaled_adj.sum(axis=1)
    non_zero_rows = row_sums[row_sums > 0]
    print(f"  Nodes with outgoing edges: {len(non_zero_rows)}/{len(scaled_adj)}")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("MULTI-GRAPH WITH LAYER TRANSITIONS - FINAL STATISTICS")
    print("=" * 80)
    basis_count = sum(1 for node in scaled_adj.index if node in basis_names)
    gene_count = scaled_adj.shape[0] - basis_count
    print(f"Total nodes: {scaled_adj.shape[0]} ({gene_count} genes + {basis_count} merged basis vectors)")
    print(f"Total possible edges: {scaled_adj.shape[0] * scaled_adj.shape[1]}")
    print(f"Non-zero edges: {(scaled_adj != 0).sum().sum()}")
    print(f"Edge density: {(scaled_adj != 0).sum().sum() / (scaled_adj.shape[0] * scaled_adj.shape[1]):.4f}")
    print(f"\nEdge weight statistics (scaled, before final normalization):")
    print(f"  Min (non-zero): {scaled_adj[scaled_adj > 0].min().min():.6f}")
    print(f"  Max: {scaled_adj.values.max():.6f}")
    print(f"  Mean: {scaled_adj.values.mean():.6f}")
    print(f"  Median: {np.median(scaled_adj.values):.6f}")
    
    # Diagnostic: check start and end node connectivity
    print(f"\nStart node '{start}' connectivity check:")
    if start in scaled_adj.index:
        start_out = (scaled_adj.loc[start, :] > 0).sum()
        start_in = (scaled_adj.loc[:, start] > 0).sum()
        print(f"  Outgoing edges: {start_out}, Incoming edges: {start_in}")
        if start_out > 0:
            print(f"  Top outgoing targets: {scaled_adj.loc[start, :].nlargest(3).to_dict()}")
    
    print(f"\nEnd node '{end}' connectivity check:")
    if end in scaled_adj.index:
        end_out = (scaled_adj.loc[end, :] > 0).sum()
        end_in = (scaled_adj.loc[:, end] > 0).sum()
        print(f"  Outgoing edges: {end_out}, Incoming edges: {end_in}")
        if end_in > 0:
            print(f"  Top incoming sources: {scaled_adj.loc[:, end].nlargest(3).to_dict()}")
    
    return scaled_adj

def create_shadow_network_multigraph(
    std_data_dict,
    intra_graphs,
    sign_matrices,
    edge_fn_inter,
    gamma=0.01,
    alpha=0.1,
    start='e3',
    end='e5'
):
    """
    Create a multi-layer graph with shadow network for negative correlations.
    OPTIMIZED VERSION: Uses single graph per layer with sign matrices.
    
    Instead of generating separate positive/negative graphs, this uses:
    - Single intra_graphs based on |correlation|
    - sign_matrices to determine positive (+1) vs negative (-1) correlations
    - Vectorized operations for edge assignment
    
    Parameters:
    -----------
    std_data_dict : dict
        Dictionary of standardized datasets
    intra_graphs : dict
        Dictionary of intra-layer adjacency matrices (based on |correlation|)
    sign_matrices : dict
        Dictionary of correlation sign matrices (+1, -1, 0) for each layer
    edge_fn_inter : function
        Function for inter-layer edge weights
    gamma : float
        Deprecated. Kept for backward compatibility and ignored.
    alpha : float
        Probability of transitioning to a different layer at each step (0 <= alpha <= 1)
    start, end : str
        Start and end basis vectors
    
    Returns:
    --------
    pd.DataFrame : Complete adjacency matrix with regular + shadow nodes
    """

    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    print("=" * 80)
    print("CREATING SHADOW NETWORK (BLOCKWISE CONSTRUCTION)")
    print("=" * 80)
    print("Building normalized regular multigraph, then applying sign masks blockwise")
    if gamma is not None:
        print(f"Note: gamma={gamma} is ignored in dynamic-shadow mode")
    print(f"Alpha (inter-layer transition probability): {alpha}")
    print(f"Intra-layer probability: {1 - alpha}")

    basis_names = ['e1', 'e2', 'e3', 'e4', 'e5']

    print("\n[STEP 1] Building normalized regular multigraph...")
    regular_adj = create_multigraph_with_layer_transitions(
        std_data_dict=std_data_dict,
        intra_layer_graphs=intra_graphs,
        edge_fn_inter=edge_fn_inter,
        alpha=alpha,
        start=start,
        end=end,
    )

    print("\n[STEP 2] Building sign graph and masks...")

    def _raw_corr(x, y):
        corr = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(corr) else corr

    signed_intra_graphs = {}
    for layer_name, adj_df in intra_graphs.items():
        layer_signs = sign_matrices[layer_name].reindex(index=adj_df.index, columns=adj_df.columns).fillna(0.0)
        signed_intra_graphs[layer_name] = adj_df * layer_signs

    sign_adj = create_multigraph(
        std_data_dict=std_data_dict,
        intra_layer_graphs=signed_intra_graphs,
        edge_fn_inter=_raw_corr,
        start=start,
        end=end,
    )
    sign_adj = np.sign(sign_adj).reindex(index=regular_adj.index, columns=regular_adj.columns).fillna(0.0)

    pos_adj = regular_adj.where(sign_adj > 0, 0.0)
    neg_adj = regular_adj.where(sign_adj < 0, 0.0)

    regular_nodes = regular_adj.index.tolist()
    non_basis_nodes = [node for node in regular_nodes if node not in basis_names]
    shadow_nodes = [f"{node}_neg" for node in non_basis_nodes]
    all_nodes = regular_nodes + shadow_nodes

    print("\n[STEP 3] Assembling blockwise shadow adjacency...")
    complete_adj = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)

    # Regular -> Regular (positive)
    complete_adj.loc[regular_nodes, regular_nodes] = pos_adj.values

    # Regular -> Shadow (negative)
    complete_adj.loc[regular_nodes, shadow_nodes] = neg_adj.loc[regular_nodes, non_basis_nodes].values

    # Shadow -> Shadow (positive)
    complete_adj.loc[shadow_nodes, shadow_nodes] = pos_adj.loc[non_basis_nodes, non_basis_nodes].values

    # Shadow -> Regular (negative)
    complete_adj.loc[shadow_nodes, non_basis_nodes] = neg_adj.loc[non_basis_nodes, non_basis_nodes].values

    # Step 4: Final statistics
    print("\n" + "=" * 80)
    print("SHADOW NETWORK - FINAL STATISTICS")
    print("=" * 80)
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Non-zero edges: {(complete_adj != 0).sum().sum()}")
    print(f"Edge density: {(complete_adj != 0).sum().sum() / (len(all_nodes) ** 2):.6f}")
    print(f"\nProbability parameters:")
    print("  P(negative jump from a node): sum(negative edge weights) / sum(total edge weights)")
    print("  P(positive stay in current sign-space): complementary per-node proportion")

    return complete_adj


def create_shadow_network_multigraph_lasso(
    std_data_dict,
    edge_fn_inter,
    gamma=0.01,
    alpha=0.1,
    start='e3',
    end='e5',
    lasso_lambda=0.01,
):
    """
    Build a shadow multi-layer graph using Lasso intra-layer edges.

    This follows the same process as create_shadow_network_multigraph:
    1) Generate Lasso intra-layer adjacency + sign matrices.
    2) Pass them into the existing shadow-network constructor.
    """
    from .generate_graph_internal import lasso_single_graph

    intra_graphs, sign_matrices = lasso_single_graph(
        std_data_dict,
        edge_fn=None,
        start=start,
        end=end,
        return_signs=True,
        lasso_lambda=lasso_lambda,
    )

    return create_shadow_network_multigraph(
        std_data_dict=std_data_dict,
        intra_graphs=intra_graphs,
        sign_matrices=sign_matrices,
        edge_fn_inter=edge_fn_inter,
        gamma=gamma,
        alpha=alpha,
        start=start,
        end=end,
    )