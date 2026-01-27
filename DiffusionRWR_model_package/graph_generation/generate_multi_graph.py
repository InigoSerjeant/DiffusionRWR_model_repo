import pandas as pd
import numpy as np

def create_connected_sections(data_A, data_B, label_A, label_B, edge_fn, same_gene=True):
    A_new_row_names = [i + label_A for i in data_A.index]
    B_new_col_names = [i + label_B for i in data_B.index]
    connected_section_array = np.zeros((len(A_new_row_names), len(B_new_col_names)))
    if same_gene:
        for i, A in enumerate(data_A.index):
            for j, B in enumerate(data_B.index):
                if A.split('*')[0] in B:
                    connected_section_array[i,j] = edge_fn(data_A.loc[A], data_B.loc[B])
    else:
        for i, A in enumerate(data_A.index):
            for j, B in enumerate(data_B.index):
                connected_section_array[i,j] = edge_fn(data_A.loc[A], data_B.loc[B])
    connected_section = pd.DataFrame(connected_section_array, index=A_new_row_names, columns=B_new_col_names)
    return connected_section

def create_multigraph(std_data_dict, intra_layer_graphs, edge_fn_inter, start='e3', end='e5'):
    """
    Create a multi-layer graph with:
    - Intra-layer connections within each dataset
    - Inter-layer connections between datasets
    
    Parameters:
    -----------
    std_data_dict : dict
        Dictionary of standardized datasets (e.g., {'rna': rna_std, 'k9me2': k9me2_std})
    edge_fn_intra : function
        Function to compute edge weights within layers (adjacency matrix)
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
    print("\n[STEP 1] Generating intra-layer graphs...")

    
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

    # Print statistics
    print("\n" + "=" * 80)
    print("MULTI-GRAPH STATISTICS")
    print("=" * 80)
    print(f"Total nodes: {complete_adj.shape[0]}")
    print(f"Total possible edges: {complete_adj.shape[0] * complete_adj.shape[1]}")
    print(f"Non-zero edges: {(complete_adj != 0).sum().sum()}")
    print(f"Edge density: {(complete_adj != 0).sum().sum() / (complete_adj.shape[0] * complete_adj.shape[1]):.4f}")
    print(f"\nEdge weight statistics:")
    print(f"  Min: {complete_adj.values.min():.6f}")
    print(f"  Max: {complete_adj.values.max():.6f}")
    print(f"  Mean: {complete_adj.values.mean():.6f}")
    print(f"  Median: {np.median(complete_adj.values):.6f}")
    return complete_adj