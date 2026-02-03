import numpy as np
import pandas as pd

def generate_single_layer_graphs(std_data_dict, edge_fn, start = 'e3', end = 'e5', return_signs=False):
    """
    Generate single-layer graph adjacency matrix using specified edge function.
    
    Parameters:
    -----------
    std_data_dict : dict
        Dictionary of standardized datasets
    edge_fn : function
        Edge weight function (can return tuple of (weights, signs) or just weights)
    start, end : str
        Start and end basis vectors
    return_signs : bool
        If True, also returns sign matrices for each layer
    
    Returns:
    --------
    adjacency_df_dict : dict
        Dictionary of adjacency matrices
    sign_matrix_dict : dict (optional, if return_signs=True)
        Dictionary of sign matrices for tracking positive/negative correlations
    """
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
    sign_matrix_dict = {} if return_signs else None
    
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
        
        # Call edge function - may return tuple or single value
        result = edge_fn(data_with_basis)
        if isinstance(result, tuple):
            adjacency_df, sign_matrix = result
            if return_signs:
                sign_matrix_dict[data_label] = sign_matrix
        else:
            adjacency_df = result
        
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
    
    if return_signs:
        return adjacency_df_dict, sign_matrix_dict
    else:
        return adjacency_df_dict


def generate_negative_correlation_graphs(std_data_dict, neg_edge_fn, start='e3', end='e5', threshold=-0.3, sigma=0.5):
    """
    Generate negative correlation adjacency matrices for each dataset.
    OPTIMIZED: Uses vectorized correlation computation instead of nested loops.
    
    Parameters:
    -----------
    std_data_dict : dict
        Dictionary of standardized datasets
    neg_edge_fn : function
        Function to compute negative correlation edge weights (used for parameter extraction only)
    start, end : str
        Start and end basis vectors (not used for negative correlations)
    threshold : float
        Correlation threshold for considering edges as negative
    sigma : float
        Gaussian kernel width parameter
    
    Returns:
    --------
    dict : Dictionary of negative correlation adjacency matrices
    """
    print(f"\nGenerating negative correlation graphs (VECTORIZED)")
    print(f"Negative correlation threshold: {threshold}")
    print(f"Gaussian sigma: {sigma}")
    
    negative_adj_dict = {}
    
    for data_label, data in std_data_dict.items():
        print(f"\n--- Processing dataset: {data_label} ---")
        print(f"Data shape: {data.shape}")
        
        # VECTORIZED: Compute all correlations at once
        corr_matrix = data.T.corr()
        
        # Extract negative correlations below threshold
        neg_mask = corr_matrix < threshold
        
        # Apply Gaussian kernel to negative correlations
        abs_neg_corr = np.abs(corr_matrix.values)
        neg_adj = np.exp(-0.5 * ((1 - abs_neg_corr) / sigma) ** 2)
        
        # Zero out positive correlations and self-loops
        neg_adj = neg_adj * neg_mask.values
        np.fill_diagonal(neg_adj, 0)
        
        # Convert to DataFrame
        neg_adj_df = pd.DataFrame(neg_adj, index=data.index, columns=data.index)
        negative_adj_dict[data_label] = neg_adj_df
        
        # Print statistics
        non_zero_edges = (neg_adj_df > 0).sum().sum()
        print(f"Negative correlation edges: {non_zero_edges}")
        if non_zero_edges > 0:
            print(f"  Min (non-zero): {neg_adj_df[neg_adj_df > 0].min().min():.6f}")
            print(f"  Max: {neg_adj_df.values.max():.6f}")
            print(f"  Mean (non-zero): {neg_adj_df[neg_adj_df > 0].mean().mean():.6f}")
    
    return negative_adj_dict