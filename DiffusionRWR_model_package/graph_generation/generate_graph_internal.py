import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

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

def lasso_single_graph(std_data_dict, edge_fn, start = 'e3', end = 'e5', return_signs=False, lasso_alpha=0.01):
    """
    Generate single-layer graph adjacency matrix using Lasso regressions.

    For each target node, we regress its 5-point trajectory against all other
    nodes in the same layer. Coefficients define incoming edges to the target.
    """
    print("\nGenerating single-layer graph using Lasso edge construction")

    n_features_mg = 5  # 5 time points
    standard_basis_mg = np.eye(n_features_mg)

    standard_basis_std_mg = np.zeros_like(standard_basis_mg)
    for i in range(n_features_mg):
        row_mean = standard_basis_mg[i].mean()
        row_std = standard_basis_mg[i].std()
        if row_std == 0:
            standard_basis_std_mg[i] = standard_basis_mg[i] - row_mean
        else:
            standard_basis_std_mg[i] = (standard_basis_mg[i] - row_mean) / row_std

    # Lasso regularization (lower = more edges, higher = sparser)
    lasso_alpha = float(lasso_alpha)
    max_iter = 10000
    tol = 1e-3  # Relaxed from 1e-6 (underdetermined system with 5 samples)
    print(f"Lasso regularization alpha: {lasso_alpha}")
    

    adjacency_df_dict = {}
    sign_matrix_dict = {} if return_signs else None

    for data_label, data in std_data_dict.items():
        print(f"\n--- Processing dataset: {data_label} ---")
        print(f"Data shape (genes x time): {data.shape}")

        # Sort inputs for deterministic node and time ordering
        data_sorted = data.sort_index().sort_index(axis=1)

        # Add basis vectors so diffusion start/end basis handling remains consistent
        basis_df = pd.DataFrame(
            standard_basis_std_mg,
            index=['e1', 'e2', 'e3', 'e4', 'e5'],
            columns=data_sorted.columns
        )
        data_with_basis = pd.concat([data_sorted, basis_df], axis=0)
        data_with_basis = data_with_basis.sort_index()

        node_names = data_with_basis.index.tolist()
        node_matrix = data_with_basis.values.astype(float)  # (n_nodes, 5)
        n_nodes = node_matrix.shape[0]

        adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
        sign_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

        for target_idx in range(n_nodes):
            predictor_mask = np.ones(n_nodes, dtype=bool)
            predictor_mask[target_idx] = False

            X = node_matrix[predictor_mask, :].T  # (5, n_nodes-1)
            y = node_matrix[target_idx, :]        # (5,)

            lasso_model = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=max_iter, tol=tol, selection='cyclic')
            lasso_model.fit(X, y)
            coefficients = lasso_model.coef_
            coefficients[np.abs(coefficients) < 1e-12] = 0.0

            predictor_indices = np.where(predictor_mask)[0]
            for coef_idx, source_idx in enumerate(predictor_indices):
                coefficient = coefficients[coef_idx]
                adjacency[source_idx, target_idx] = np.abs(coefficient)
                sign_matrix[source_idx, target_idx] = int(np.sign(coefficient))

        adjacency_df = pd.DataFrame(adjacency, index=node_names, columns=node_names)
        sign_df = pd.DataFrame(sign_matrix, index=node_names, columns=node_names)
        adjacency_df = adjacency_df.sort_index().sort_index(axis=1)
        sign_df = sign_df.sort_index().sort_index(axis=1)

        # Remove basis vectors except selected start/end, consistent with other generators
        if start in ['e1', 'e2', 'e3', 'e4', 'e5'] or end in ['e1', 'e2', 'e3', 'e4', 'e5']:
            nodes_to_zero = [basis for basis in ['e1', 'e2', 'e3', 'e4', 'e5'] if basis not in [start, end]]
            for basis in nodes_to_zero:
                if basis in adjacency_df.index:
                    adjacency_df.loc[basis, :] = 0
                    adjacency_df.loc[:, basis] = 0
                    sign_df.loc[basis, :] = 0
                    sign_df.loc[:, basis] = 0

        nonzero_matrix = adjacency_df.values > 0
        out_counts = nonzero_matrix.sum(axis=1)
        in_counts = nonzero_matrix.sum(axis=0)
        ordered_nodes = adjacency_df.index.tolist()

        max_out = int(out_counts.max())
        min_out = int(out_counts.min())
        max_in = int(in_counts.max())
        min_in = int(in_counts.min())

        max_out_nodes = [ordered_nodes[idx] for idx, count in enumerate(out_counts) if count == max_out]
        min_out_nodes = [ordered_nodes[idx] for idx, count in enumerate(out_counts) if count == min_out]
        max_in_nodes = [ordered_nodes[idx] for idx, count in enumerate(in_counts) if count == max_in]
        min_in_nodes = [ordered_nodes[idx] for idx, count in enumerate(in_counts) if count == min_in]

        print(f"Highest number of outgoing edges: {max_out} (nodes: {max_out_nodes[0]})")
        print(f"Lowest number of outgoing edges: {min_out} (nodes: {min_out_nodes[0]})")
        print(f"Highest number of incoming edges: {max_in} (nodes: {max_in_nodes[0]})")
        print(f"Lowest number of incoming edges: {min_in} (nodes: {min_in_nodes[0]})")
        print(f"Adjacency matrix shape: {adjacency_df.shape}")
        print(f"Lasso regularization alpha: {lasso_alpha}")

        adjacency_df_dict[data_label] = adjacency_df
        if return_signs:
            sign_matrix_dict[data_label] = sign_df

    if return_signs:
        return adjacency_df_dict, sign_matrix_dict
    return adjacency_df_dict
