import numpy as np
import pandas as pd

def generate_single_layer_graphs(std_data_dict, edge_fn, start = 'e3', end = 'e5'):
    """Generate single-layer graph adjacency matrix using specified edge function."""
    print(f"\nGenerating single-layer graph using edge function: {edge_fn.__name__}")

    n_features_mg = 5  # 5 time points
    standard_basis_mg = np.eye(n_features_mg)

        # Standardize basis vectors row-wise
    standard_basis_std_mg = np.zeros_like(standard_basis_mg)
    for i in range(n_features_mg):
        row_mean = standard_basis_mg[i].mean()
        row_std = standard_basis_mg[i].std()
        if row_std == 0:
            # Handle constant rows (shouldn't happen with identity matrix, but just in case)
            standard_basis_std_mg[i] = standard_basis_mg[i] - row_mean
        else:
            standard_basis_std_mg[i] = (standard_basis_mg[i] - row_mean) / row_std

    adjacency_df_dict = {}
    for data_label, data in std_data_dict.items():
        print(f"\n--- Processing dataset: {data_label} ---")
        print(f"Data columns: {data.columns.tolist()}")
        for i, label in enumerate(['e1', 'e2', 'e3', 'e4', 'e5']):
            print(f"Basis vector {label}: {standard_basis_std_mg[i]}")
        
        # Use the same column names as the data
        data_with_basis = pd.concat([data, pd.DataFrame(standard_basis_std_mg, 
                                                        index=['e1', 'e2', 'e3', 'e4', 'e5'], 
                                                        columns=data.columns)], axis=0)
        print(f"Data shape with basis vectors: {data_with_basis.shape} ({data.shape[0]} genes + 5 basis vectors)")
        adjacency_df = edge_fn(data_with_basis)
        adjacency_df_dict[data_label] = adjacency_df


    if start in ['e1', 'e2', 'e3', 'e4', 'e5'] or end in ['e1', 'e2', 'e3', 'e4', 'e5']:
        # zero weights for all bbasis vectors except start and end
        nodes_to_zero = [i for i in ['e1', 'e2', 'e3', 'e4', 'e5'] if i not in [start, end]]
        for i in nodes_to_zero:
            print(f"\nRemoving edges for basis vector: {i}")
            for adjacency_df in adjacency_df_dict.values():
                if i in adjacency_df.index:
                    adjacency_df.loc[i, :] = 0
                    adjacency_df.loc[:, i] = 0
    for data_label, adjacency_df in adjacency_df_dict.items():
        print(f"Adjacency matrix shape: {adjacency_df.shape}")
        print(f"\nAdjacency matrix statistics ({edge_fn.__name__}):")
        print(f"  Min: {adjacency_df.values.min():.6f}")
        print(f"  Max: {adjacency_df.values.max():.6f}")
        print(f"  Mean: {adjacency_df.values.mean():.6f}")
        print(f"  Median: {np.median(adjacency_df.values):.6f}")
    
    return adjacency_df_dict