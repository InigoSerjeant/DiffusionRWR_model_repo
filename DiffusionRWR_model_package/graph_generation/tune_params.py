from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
from functools import partial
from pathlib import Path

from DiffusionRWR_model_package.data_sorting import std_data_dict
from DiffusionRWR_model_package.graph_generation.generate_graph_internal import lasso_single_graph
from DiffusionRWR_model_package.graph_generation.generate_graph_internal import generate_single_layer_graphs
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import cor_gaussian_abs
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import cor_exponential_abs
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import corr_power
from DiffusionRWR_model_package.graph_generation.lasso_SPD import proximal_precision, f


def _get_pywgcna_class():
    try:
        from PyWGCNA.wgcna import WGCNA
    except Exception as exc:
        raise ImportError("pyWGCNA is required. Install with: pip install pyWGCNA") from exc
    return WGCNA

def mean_degree_from_adjacency(adjacency: pd.DataFrame | np.ndarray) -> float:
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=True)
    else:
        matrix = np.asarray(adjacency, dtype=float).copy()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    np.fill_diagonal(matrix, 0.0)
    k = matrix.sum(axis=1)
    return float(np.mean(k[np.isfinite(k)]))

def clustering_coefficient_from_adjacency(adjacency: pd.DataFrame | np.ndarray) -> float:
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=True)
    else:
        matrix = np.asarray(adjacency, dtype=float).copy()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    # Use absolute edge weights throughout this metric.
    matrix = np.abs(matrix)
    np.fill_diagonal(matrix, 0.0)

    # Global maximum off-diagonal edge weight for normalization.
    n = matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    w_max = float(np.max(matrix[off_diag_mask])) if np.any(off_diag_mask) else 0.0
    if w_max <= 0.0:
        return 0.0

    # Equation implemented (weighted Zhang-style):
    # C_i = (1 / w_max) * [ sum_{j,l != i, j != l} w_ij w_jl w_li ]
    #                      / [ sum_{j,l != i, j != l} w_ij w_li ]
    clustering_coeffs = np.zeros(n, dtype=float)
    for i in range(n):
        num = 0.0
        den = 0.0
        for j in range(n):
            if j == i:
                continue
            w_ij = matrix[i, j]
            if w_ij <= 0.0:
                continue
            for l in range(n):
                if l == i or l == j:
                    continue
                w_li = matrix[l, i]
                if w_li <= 0.0:
                    continue
                num += w_ij * matrix[j, l] * w_li
                den += w_ij * w_li

        if den > 0.0:
            clustering_coeffs[i] = (num / den) / w_max

    finite = clustering_coeffs[np.isfinite(clustering_coeffs)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))

def number_isolated_nodes_from_adjacency(adjacency: pd.DataFrame | np.ndarray) -> int:
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=True)
    else:
        matrix = np.asarray(adjacency, dtype=float).copy()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    np.fill_diagonal(matrix, 0.0)
    k = matrix.sum(axis=1)
    return int(np.sum(np.isclose(k, 0.0, atol=0.01)))

def _connectivity_from_adjacency(adjacency: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=True)
    else:
        matrix = np.asarray(adjacency, dtype=float).copy()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    np.fill_diagonal(matrix, 0.0)
    k = matrix.sum(axis=1)
    return k[np.isfinite(k)]


def _adjacency_sparsity(adjacency: pd.DataFrame | np.ndarray) -> float:
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=False)
    else:
        matrix = np.asarray(adjacency, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    n = matrix.shape[0]
    if n <= 1:
        return 0.0

    off_diag = ~np.eye(n, dtype=bool)
    off_diag_values = matrix[off_diag]
    return float(np.mean(np.isclose(off_diag_values, 0.0, atol=0.01)))


def _save_current_plot(output_dir, filename):
    if output_dir is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / filename, dpi=200, bbox_inches="tight")


def wgcna_scale_invariance_r2(connectivity: np.ndarray, n_bins: int = 10) -> dict[str, float | int]:
    WGCNA = _get_pywgcna_class()

    values = np.asarray(connectivity, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size < 3:
        return {
            "Sign of Slope": np.nan,
            "r2": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "n_points": 0,
        }

    fit_df = WGCNA.scaleFreeFitIndex(k=values.tolist(), nBreaks=int(n_bins))
    r2 = float(fit_df.loc[0, "Rsquared.SFT"])
    slope = float(fit_df.loc[0, "slope.SFT"])

    return {
        "Sign of Slope": float(np.sign(slope)),
        "r2": r2,
        "slope": slope,
        "intercept": np.nan,
        "n_points": int(values.size),
    }


def wgcna_scale_invariance_from_adjacency(
    adjacency: pd.DataFrame | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | int]:
    connectivity = _connectivity_from_adjacency(adjacency)
    return wgcna_scale_invariance_r2(connectivity=connectivity, n_bins=n_bins)


def plot_scale_invariance_vs_hyperparameter(graph_dataframe_gen_function, hyperparameter_values, hyperparameter_name, output_dir=None):
    r2_values = []
    sparsity_values = []
    for value in hyperparameter_values:
        adjacency = graph_dataframe_gen_function(value)
        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        r2_values.append(metrics["r2"])
        sparsity_values.append(_adjacency_sparsity(adjacency))

    plt.figure(figsize=(8, 6))
    plt.plot(hyperparameter_values, r2_values, marker="o")
    plt.title(f"Scale Invariance R² vs {hyperparameter_name}")
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, f"scale_invariance_vs_{hyperparameter_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(hyperparameter_values, sparsity_values, marker="o")
    plt.title(f"Adjacency Sparsity vs {hyperparameter_name}")
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_{hyperparameter_name}.png")


def plot_scale_inv_lasso_lambda(lambda_values, output_dir=None):
    scale_invars = []
    sparsity_values = []

    lasso_param_name = "lasso_alpha"
    try:
        params = inspect.signature(lasso_single_graph).parameters
        if "lasso_lambda" in params:
            lasso_param_name = "lasso_lambda"
        elif "lasso_alpha" in params:
            lasso_param_name = "lasso_alpha"
    except Exception:
        pass

    for lambda_value in lambda_values:
        adjacency_dict = lasso_single_graph(
            std_data_dict,
            edge_fn=None,
            start="e3",
            end="e5",
            return_signs=False,
            **{lasso_param_name: lambda_value},
        )

        if "rna_ai" in adjacency_dict:
            adjacency = adjacency_dict["rna_ai"]
        else:
            adjacency = next(iter(adjacency_dict.values()))

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics["r2"])
        sparsity_values.append(_adjacency_sparsity(adjacency))

        print(
            f"Lambda: {lambda_value}, R²: {metrics['r2']:.4f}, "
            f"Slope: {metrics['slope']:.4f}, Sign of Slope: {metrics['Sign of Slope']}"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, scale_invars, marker="o")
    plt.title("Scale Invariance R² vs Lasso Lambda")
    plt.xlabel("Lasso Lambda")
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, "scale_invariance_vs_lasso_lambda.png")

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Lasso Lambda")
    plt.xlabel("Lasso Lambda")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, "adjacency_sparsity_vs_lasso_lambda.png")


def plot_scale_inv_spd_lasso_lambda(
    lambda_values,
    dataset_name="rna_ai",
    top_n_genes=None,
    gamma=0.01,
    eta=0.5,
    max_iter=100,
    output_dir=None,
):
    scale_invars = []
    sparsity_values = []
    isolated_nodes_values = []
    clustering_coefficients = []

    if dataset_name not in std_data_dict:
        raise KeyError(f"Dataset '{dataset_name}' not found in std_data_dict")

    data_df = std_data_dict[dataset_name].copy().dropna(axis=0, how="any")
    if top_n_genes is not None and int(top_n_genes) > 0 and len(data_df) > int(top_n_genes):
        top_idx = data_df.var(axis=1).sort_values(ascending=False).head(int(top_n_genes)).index
        data_df = data_df.loc[top_idx]

    if data_df.empty:
        raise ValueError(f"Dataset '{dataset_name}' is empty after preprocessing")

    X = data_df.T.to_numpy(dtype=float)

    for lambda_value in lambda_values:
        Q, _ = proximal_precision(
            X,
            lam=float(lambda_value),
            gamma=float(gamma),
            eta=float(eta),
            max_iter=int(max_iter),
        )

        adjacency = np.abs(np.asarray(Q, dtype=float))
        np.fill_diagonal(adjacency, 0.0)

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics)
        sparsity_values.append(_adjacency_sparsity(adjacency))
        isolated_nodes_values.append(number_isolated_nodes_from_adjacency(adjacency))
        clustering_coefficients.append(clustering_coefficient_from_adjacency(adjacency))

        print(
            f"Lambda: {lambda_value}, R²: {metrics['r2']:.4f}, "
            f"Slope: {metrics['slope']:.4f}, Sign of Slope: {metrics['Sign of Slope']}"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, [metrics["r2"] for metrics in scale_invars], marker="o")
    plt.title("Scale Invariance R² vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, f"scale_invariance_vs_spd_lasso_lambda_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_spd_lasso_lambda_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(lambda_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_spd_lasso_lambda_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, isolated_nodes_values, marker="o")
    plt.title("Number of Isolated Nodes vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Number of Isolated Nodes")
    plt.grid()
    _save_current_plot(output_dir, f"isolated_nodes_vs_spd_lasso_lambda_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, clustering_coefficients, marker="o")
    plt.title("Clustering Coefficient vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Mean Clustering Coefficient")
    plt.grid()
    _save_current_plot(output_dir, f"clustering_coefficient_vs_spd_lasso_lambda_{dataset_name}.png")


def plot_lasso_lambda_least_squares(
    lambda_values,
    dataset_name="rna_ai",
    top_n_genes=None,
    gamma=0.01,
    eta=0.5,
    max_iter=100,
    output_dir=None,
):
    """Plot least squares loss f(omega) for SPD-Lasso across lambda values."""
    if dataset_name not in std_data_dict:
        raise KeyError(f"Dataset '{dataset_name}' not found in std_data_dict")

    data_df = std_data_dict[dataset_name].copy().dropna(axis=0, how="any")
    if top_n_genes is not None and int(top_n_genes) > 0 and len(data_df) > int(top_n_genes):
        top_idx = data_df.var(axis=1).sort_values(ascending=False).head(int(top_n_genes)).index
        data_df = data_df.loc[top_idx]

    if data_df.empty:
        raise ValueError(f"Dataset '{dataset_name}' is empty after preprocessing")

    X = data_df.T.to_numpy(dtype=float)
    ls_losses = []

    for lambda_value in lambda_values:
        omega, _ = proximal_precision(
            X,
            lam=float(lambda_value),
            gamma=float(gamma),
            eta=float(eta),
            max_iter=int(max_iter),
            return_omega=True,
            show_ls_loss=False,
        )

        tau = 1.0 / np.diag(omega)
        ls_loss = f(omega, X, tau)
        ls_losses.append(ls_loss)

        print(f"Lambda: {lambda_value}, LS loss: {ls_loss:.6f}")

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, ls_losses, marker="o")
    plt.title("Least Squares Loss vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Least Squares Loss f(omega)")
    plt.grid()
    _save_current_plot(output_dir, f"ls_loss_vs_spd_lasso_lambda_{dataset_name}.png")


def plot_scale_inv_single_layer_sigma_gaussian(sigma_values, dataset_name="rna_ai", start="e3", end="e5", output_dir=None):
    scale_invars = []
    sparsity_values = []
    isolated_nodes_values = []
    clustering_coefficients = []

    if dataset_name not in std_data_dict:
        raise KeyError(f"Dataset '{dataset_name}' not found in std_data_dict")

    # Preserve the exact mathematics for the selected dataset while avoiding
    # redundant graph generation for all other datasets in each sigma step.
    selected_data_dict = {dataset_name: std_data_dict[dataset_name]}

    for sigma_value in sigma_values:
        edge_fn = partial(cor_gaussian_abs, sigma=sigma_value)
        edge_fn.__name__ = f"cor_gaussian_abs_sigma_{sigma_value}"

        adjacency_dict = generate_single_layer_graphs(
            selected_data_dict,
            edge_fn=edge_fn,
            start=start,
            end=end,
            return_signs=False,
        )

        adjacency = adjacency_dict[dataset_name]

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics)
        sparsity_values.append(_adjacency_sparsity(adjacency))
        isolated_nodes_values.append(number_isolated_nodes_from_adjacency(adjacency))
        clustering_coefficients.append(clustering_coefficient_from_adjacency(adjacency))

        print(
            f"Sigma: {sigma_value}, R²: {metrics['r2']:.4f}, "
            f"Slope: {metrics['slope']:.4f}, Sign of Slope: {metrics['Sign of Slope']}"
        )
    #Plot mean degree as well to check we're not just losing connectivity
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, [metrics["n_points"] for metrics in scale_invars
], marker="o")
    plt.title("Number of Points (Non-zero Connectivity) vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Number of Points with Non-zero Connectivity")
    plt.grid()
    _save_current_plot(output_dir, f"nonzero_connectivity_vs_sigma_gaussian_{dataset_name}.png")
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, [metrics["r2"] for metrics in scale_invars], marker="o")
    plt.title("Scale Invariance R² vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, f"scale_invariance_vs_sigma_gaussian_{dataset_name}.png")
    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(sigma_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_sigma_gaussian_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_sigma_gaussian_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, isolated_nodes_values, marker="o")
    plt.title("Number of Isolated Nodes vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Number of Isolated Nodes")
    plt.grid()
    _save_current_plot(output_dir, f"isolated_nodes_vs_sigma_gaussian_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, clustering_coefficients, marker="o")
    plt.title("Clustering Coefficient vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Mean Clustering Coefficient")
    plt.grid()
    _save_current_plot(output_dir, f"clustering_coefficient_vs_sigma_gaussian_{dataset_name}.png")


def plot_scale_inv_single_layer_sigma_exponential(sigma_values, dataset_name="rna_ai", start="e3", end="e5", output_dir=None):
    scale_invars = []
    sparsity_values = []
    isolated_nodes_values = []
    clustering_coefficients = []

    if dataset_name not in std_data_dict:
        raise KeyError(f"Dataset '{dataset_name}' not found in std_data_dict")

    # Preserve the exact mathematics for the selected dataset while avoiding
    # redundant graph generation for all other datasets in each sigma step.
    selected_data_dict = {dataset_name: std_data_dict[dataset_name]}

    for sigma_value in sigma_values:
        edge_fn = partial(cor_exponential_abs, sigma=sigma_value)
        edge_fn.__name__ = f"cor_exponential_abs_sigma_{sigma_value}"

        adjacency_dict = generate_single_layer_graphs(
            selected_data_dict,
            edge_fn=edge_fn,
            start=start,
            end=end,
            return_signs=False,
        )

        adjacency = adjacency_dict[dataset_name]

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics)
        sparsity_values.append(_adjacency_sparsity(adjacency))
        isolated_nodes_values.append(number_isolated_nodes_from_adjacency(adjacency))
        clustering_coefficients.append(clustering_coefficient_from_adjacency(adjacency))

        print(
            f"Sigma: {sigma_value}, R²: {metrics['r2']:.4f}, "
            f"Slope: {metrics['slope']:.4f}, Sign of Slope: {metrics['Sign of Slope']}"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, [metrics["r2"] for metrics in scale_invars], marker="o")
    plt.title("Scale Invariance R² vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, f"scale_invariance_vs_sigma_exponential_{dataset_name}.png")
    #Plot sign on slope as well
    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(sigma_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_sigma_exponential_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_sigma_exponential_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, isolated_nodes_values, marker="o")
    plt.title("Number of Isolated Nodes vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Number of Isolated Nodes")
    plt.grid()
    _save_current_plot(output_dir, f"isolated_nodes_vs_sigma_exponential_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, clustering_coefficients, marker="o")
    plt.title("Clustering Coefficient vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Mean Clustering Coefficient")
    plt.grid()
    _save_current_plot(output_dir, f"clustering_coefficient_vs_sigma_exponential_{dataset_name}.png")


def plot_scale_inv_single_layer_power(power_values, dataset_name="rna_ai", start="e3", end="e5", output_dir=None):
    scale_invars = []
    sparsity_values = []
    isolated_nodes_values = []
    clustering_coefficients = []

    if dataset_name not in std_data_dict:
        raise KeyError(f"Dataset '{dataset_name}' not found in std_data_dict")

    # Preserve the exact mathematics for the selected dataset while avoiding
    # redundant graph generation for all other datasets in each power step.
    selected_data_dict = {dataset_name: std_data_dict[dataset_name]}

    for n_power in power_values:
        edge_fn = partial(corr_power, n_power=n_power)
        edge_fn.__name__ = f"corr_power_n_{n_power}"

        adjacency_dict = generate_single_layer_graphs(
            selected_data_dict,
            edge_fn=edge_fn,
            start=start,
            end=end,
            return_signs=False,
        )

        adjacency = adjacency_dict[dataset_name]

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics)
        sparsity_values.append(_adjacency_sparsity(adjacency))
        isolated_nodes_values.append(number_isolated_nodes_from_adjacency(adjacency))
        clustering_coefficients.append(clustering_coefficient_from_adjacency(adjacency))

        print(
            f"Power: {n_power}, R²: {metrics['r2']:.4f}, "
            f"Slope: {metrics['slope']:.4f}, Sign of Slope: {metrics['Sign of Slope']}"
        )

    plt.figure(figsize=(8, 6))
    plt.plot(power_values, [metrics["r2"] for metrics in scale_invars], marker="o")
    plt.title("Scale Invariance R² vs Correlation Power (Single Layer)")
    plt.xlabel("Correlation Power")
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, f"scale_invariance_vs_corr_power_{dataset_name}.png")

    # Plot sign of slope as well.
    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(power_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs Correlation Power (Single Layer)")
    plt.xlabel("Correlation Power")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_corr_power_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(power_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Correlation Power (Single Layer)")
    plt.xlabel("Correlation Power")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_corr_power_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(power_values, isolated_nodes_values, marker="o")
    plt.title("Number of Isolated Nodes vs Correlation Power (Single Layer)")
    plt.xlabel("Correlation Power")
    plt.ylabel("Number of Isolated Nodes")
    plt.grid()
    _save_current_plot(output_dir, f"isolated_nodes_vs_corr_power_{dataset_name}.png")

    plt.figure(figsize=(8, 6))
    plt.plot(power_values, clustering_coefficients, marker="o")
    plt.title("Clustering Coefficient vs Correlation Power (Single Layer)")
    plt.xlabel("Correlation Power")
    plt.ylabel("Mean Clustering Coefficient")
    plt.grid()
    _save_current_plot(output_dir, f"clustering_coefficient_vs_corr_power_{dataset_name}.png")

if __name__ == "__main__":
    sigma_values = [0.0004 * i for i in range(1, 51)]
    power_values = list(range(10, 41))
    spd_lambda_values = [0.04*i for i in range(1, 26)]
    output_dir = Path(__file__).resolve().parent / "tuning_results"
    for i in ["rna_ai", "k20me3", "k9me2", "k27me3"]:
        #plot_scale_inv_single_layer_sigma_gaussian(
        #    sigma_values=sigma_values,
        #    dataset_name=i,
        #    start="GPX3",
        #    end="GPX3",
        #    output_dir=output_dir,
        #) # Results imply we should use sigma=0.005 for rna
        #plot_scale_inv_single_layer_sigma_exponential(
        #    sigma_values=sigma_values,
        #    dataset_name=i,
        #    start="GPX3",
        #    end="GPX3",
        #    output_dir=output_dir,
        #) # Results imply we should use sigma=0.007 for rna
        plot_scale_inv_single_layer_power(
            power_values=power_values,
            dataset_name=i,
            start="GPX3",
            end="GPX3",
            output_dir=output_dir,
        )

        #plot_lasso_lambda_least_squares(
        #    lambda_values=spd_lambda_values,
        #    dataset_name=i,
        #    top_n_genes=None,
        #    gamma=0.1,
        #    eta=5.0,
        #    max_iter=500,
        #    output_dir=output_dir,
        #)