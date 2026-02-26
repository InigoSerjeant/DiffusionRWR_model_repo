import numpy as np
import pandas as pd

def corr_power(combined_data, n_power = 30):
    corr_matrix = 0.5*(combined_data.T.corr() +1)
    return abs(corr_matrix) ** n_power

def cor_gaussian_shifted(combined_data, sigma=0.05):
    corr_matrix = 1 - 0.5*(combined_data.T.corr() +1)
    return np.exp(-0.5 * ((corr_matrix) / sigma) ** 2)

def cor_exponential_shifted(combined_data, sigma=0.05):
    corr_matrix = 1 - 0.5*(combined_data.T.corr() + 1)
    return np.exp(-(corr_matrix) / sigma)

def intra_layer_corr_gaussian_shifted(vec_a, vec_b, sigma=1):
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    return np.exp(-0.5 * ((1 - 0.5 * (correlation + 1)) / sigma) ** 2)

def intra_layer_corr_exponential_shifted(vec_a, vec_b, sigma=1):
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    return np.exp(-(1 - 0.5 * (correlation + 1)) / sigma)

def cor_gaussian_abs_inter(vec_a, vec_b, sigma=0.05):
    """
    Inter-layer edge weight using absolute correlation with Gaussian kernel.
    Takes two vectors (one from each layer) and computes the weight.
    """
    corr = np.abs(np.corrcoef(vec_a, vec_b)[0, 1])
    return np.exp(-0.5 * ((1 - corr) / sigma) ** 2)

def cor_exponential_abs_inter(vec_a, vec_b, sigma=0.05):
    """
    Inter-layer edge weight using absolute correlation with exponential kernel.
    Takes two vectors (one from each layer) and computes the weight.
    """
    corr = np.abs(np.corrcoef(vec_a, vec_b)[0, 1])
    return np.exp(-(1 - corr) / sigma)

def cor_gaussian_abs(combined_data, sigma=0.05):
    """
    Generate edge weights based on absolute correlation with Gaussian kernel.
    Also returns sign matrix for tracking positive/negative correlations.
    
    Weight function: exp(-0.5 * ((1 - |correlation|) / sigma)^2)
    
    Parameters:
    -----------
    combined_data : pd.DataFrame
        Data with genes and basis vectors
    sigma : float
        Gaussian kernel width parameter (default 0.5)
    
    Returns:
    --------
    adjacency_df : pd.DataFrame
        Edge weights based on absolute correlation
    sign_matrix : pd.DataFrame
        Sign of correlations (+1, -1, or 0)
    """
    # Compute correlation matrix once
    corr_matrix = combined_data.T.corr()
    
    # Get sign matrix for later reference
    sign_matrix = corr_matrix.apply(np.sign)
    
    # Compute edge weights using absolute correlation
    abs_corr = corr_matrix.abs()
    weights = np.exp(-0.5 * ((1 - abs_corr.values) / sigma) ** 2)
    
    # Zero diagonal (no self-loops)
    np.fill_diagonal(weights, 0)
    
    # Create DataFrame
    adjacency_df = pd.DataFrame(weights, index=corr_matrix.index, columns=corr_matrix.columns)
    
    return adjacency_df, sign_matrix

def cor_exponential_abs(combined_data, sigma=0.01):
    """
    Generate edge weights based on absolute correlation with exponential kernel.
    Also returns sign matrix for tracking positive/negative correlations.

    Weight function: exp(-(1 - |correlation|) / sigma)

    Parameters:
    -----------
    combined_data : pd.DataFrame
        Data with genes and basis vectors
    sigma : float
        Exponential kernel width parameter

    Returns:
    --------
    adjacency_df : pd.DataFrame
        Edge weights based on absolute correlation
    sign_matrix : pd.DataFrame
        Sign of correlations (+1, -1, or 0)
    """
    corr_matrix = combined_data.T.corr()
    sign_matrix = corr_matrix.apply(np.sign)
    abs_corr = corr_matrix.abs()
    weights = np.exp(-(1 - abs_corr.values) / sigma)
    np.fill_diagonal(weights, 0)
    adjacency_df = pd.DataFrame(weights, index=corr_matrix.index, columns=corr_matrix.columns)
    return adjacency_df, sign_matrix

def inter_layer_corr_power(vec_a, vec_b, n_power=20):
    """Compute edge weight between nodes in different layers based on correlation."""
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    # Convert correlation to similarity (0 to 1 range)
    similarity = 0.5 * (correlation + 1)
    # Apply power transformation
    return abs(similarity) ** n_power


def negative_correlation_weight(vec_a, vec_b, threshold=-0.3, sigma=0.05):
    """
    Compute edge weight for negatively correlated pairs using Gaussian kernel.
    
    Parameters:
    -----------
    vec_a, vec_b : array-like
        Data vectors for the two nodes
    threshold : float
        Minimum correlation value to consider (must be negative, e.g., -0.3)
    sigma : float
        Gaussian width parameter for weighting (controls sensitivity)
    
    Returns:
    --------
    float : Weight for negative correlation edge (0 if not negatively correlated enough)
    """
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]

        # Use absolute value of negative correlation
        # Apply Gaussian kernel directly on correlation magnitude
    abs_neg_corr = abs(correlation)
        # Gaussian kernel: higher correlation magnitude = higher weight
    return np.exp(-0.5 * ((1 - abs_neg_corr) / sigma) ** 2)


def negative_correlation_weight_exponential(vec_a, vec_b, threshold=-0.3, sigma=0.05):
    """
    Compute edge weight for negatively correlated pairs using exponential kernel.

    Parameters:
    -----------
    vec_a, vec_b : array-like
        Data vectors for the two nodes
    threshold : float
        Minimum correlation value to consider (must be negative, e.g., -0.3)
    sigma : float
        Exponential width parameter for weighting (controls sensitivity)

    Returns:
    --------
    float : Weight for negative correlation edge (0 if not negatively correlated enough)
    """
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    abs_neg_corr = abs(correlation)
    return np.exp(-(1 - abs_neg_corr) / sigma)


def get_correlation_sign(vec_a, vec_b):
    """
    Determine the sign of correlation between two vectors.
    
    Returns:
    --------
    int : +1 if positive correlation, -1 if negative, 0 if near zero
    """
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    if correlation > 0.1:
        return 1
    elif correlation < -0.1:
        return -1
    else:
        return 0