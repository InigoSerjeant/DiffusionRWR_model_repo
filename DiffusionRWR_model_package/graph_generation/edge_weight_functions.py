import numpy as np

def corr_power(combined_data, n_power = 30):
    corr_matrix = 0.5*(combined_data.T.corr() +1)
    return abs(corr_matrix) ** n_power

def cor_gaussian_shifted(combined_data, sigma=0.05):
    corr_matrix = 1 - 0.5*(combined_data.T.corr() +1)
    return np.exp(-0.5 * ((corr_matrix) / sigma) ** 2)

def intra_layer_corr_gaussian_shifted(vec_a, vec_b, sigma=0.1):
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    return np.exp(-0.5 * ((1 - 0.5 * (correlation + 1)) / sigma) ** 2)

def cor_gaussian_abs(combined_data, sigma=0.05):
    corr_matrix = combined_data.T.corr()
    pos_neg_corr = corr_matrix.sign()
    corr_matrix = abs(corr_matrix)
    return np.exp(-0.5 * (corr_matrix / sigma) ** 2), pos_neg_corr

def inter_layer_corr_power(vec_a, vec_b, n_power=20):
    """Compute edge weight between nodes in different layers based on correlation."""
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    # Convert correlation to similarity (0 to 1 range)
    similarity = 0.5 * (correlation + 1)
    # Apply power transformation
    return abs(similarity) ** n_power