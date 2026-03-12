from __future__ import annotations

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
from DiffusionRWR_model_package.graph_generation.lasso_SPD import proximal_precision


def _get_pywgcna_class():
    try:
        from PyWGCNA.wgcna import WGCNA
    except Exception as exc:
        raise ImportError("pyWGCNA is required. Install with: pip install pyWGCNA") from exc
    return WGCNA


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
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(hyperparameter_values, sparsity_values, marker="o")
    plt.title(f"Adjacency Sparsity vs {hyperparameter_name}")
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_{hyperparameter_name}.png")
    plt.show()


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
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Lasso Lambda")
    plt.xlabel("Lasso Lambda")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, "adjacency_sparsity_vs_lasso_lambda.png")
    plt.show()


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
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_spd_lasso_lambda_{dataset_name}.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(lambda_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs SPD-Lasso Lambda")
    plt.xlabel("SPD-Lasso Lambda")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_spd_lasso_lambda_{dataset_name}.png")
    plt.show()


def plot_scale_inv_single_layer_sigma_gaussian(sigma_values, dataset_name="rna_ai", start="e3", end="e5", output_dir=None):
    scale_invars = []
    sparsity_values = []

    for sigma_value in sigma_values:
        edge_fn = partial(cor_gaussian_abs, sigma=sigma_value)
        edge_fn.__name__ = f"cor_gaussian_abs_sigma_{sigma_value}"

        adjacency_dict = generate_single_layer_graphs(
            std_data_dict,
            edge_fn=edge_fn,
            start=start,
            end=end,
            return_signs=False,
        )

        if dataset_name in adjacency_dict:
            adjacency = adjacency_dict[dataset_name]
        else:
            adjacency = next(iter(adjacency_dict.values()))

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics)
        sparsity_values.append(_adjacency_sparsity(adjacency))

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
    plt.show()
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, [metrics["r2"] for metrics in scale_invars], marker="o")
    plt.title("Scale Invariance R² vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Scale Invariance R²")
    plt.grid()
    _save_current_plot(output_dir, f"scale_invariance_vs_sigma_gaussian_{dataset_name}.png")
    plt.show()
    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(sigma_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_sigma_gaussian_{dataset_name}.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Sigma (Single Layer, Gaussian Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_sigma_gaussian_{dataset_name}.png")
    plt.show()


def plot_scale_inv_single_layer_sigma_exponential(sigma_values, dataset_name="rna_ai", start="e3", end="e5", output_dir=None):
    scale_invars = []
    sparsity_values = []

    for sigma_value in sigma_values:
        edge_fn = partial(cor_exponential_abs, sigma=sigma_value)
        edge_fn.__name__ = f"cor_exponential_abs_sigma_{sigma_value}"

        adjacency_dict = generate_single_layer_graphs(
            std_data_dict,
            edge_fn=edge_fn,
            start=start,
            end=end,
            return_signs=False,
        )

        if dataset_name in adjacency_dict:
            adjacency = adjacency_dict[dataset_name]
        else:
            adjacency = next(iter(adjacency_dict.values()))

        metrics = wgcna_scale_invariance_from_adjacency(adjacency)
        scale_invars.append(metrics)
        sparsity_values.append(_adjacency_sparsity(adjacency))

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
    plt.show()
    #Plot sign on slope as well
    plt.figure(figsize=(8, 6))
    sign_of_slope = [np.sign(metrics["slope"]) for metrics in scale_invars]
    plt.plot(sigma_values, sign_of_slope, marker="o")
    plt.title("Sign of Slope vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Sign of Slope")
    plt.grid()
    _save_current_plot(output_dir, f"sign_of_slope_vs_sigma_exponential_{dataset_name}.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(sigma_values, sparsity_values, marker="o")
    plt.title("Adjacency Sparsity vs Sigma (Single Layer, Exponential Kernel)")
    plt.xlabel("Sigma")
    plt.ylabel("Adjacency Sparsity (Off-diagonal Zero Fraction)")
    plt.grid()
    _save_current_plot(output_dir, f"adjacency_sparsity_vs_sigma_exponential_{dataset_name}.png")
    plt.show()

if __name__ == "__main__":
    sigma_values = [0.001 * i for i in range(1, 101)]
    spd_lambda_values = [0.04*i-0.04 for i in range(1, 51)]
    output_dir = Path(__file__).resolve().parent / "outputa"
    plot_scale_inv_single_layer_sigma_gaussian(
        sigma_values=sigma_values,
        dataset_name="rna_ai",
        start="GPX3",
        end="GPX3",
        output_dir=output_dir,
    ) # Results imply we should use sigma=0.005 for rna
    plot_scale_inv_single_layer_sigma_exponential(
        sigma_values=sigma_values,
        dataset_name="rna_ai",
        start="GPX3",
        end="GPX3",
        output_dir=output_dir,
    ) # Results imply we should use sigma=0.007 for rna
    plot_scale_inv_spd_lasso_lambda(
        lambda_values=spd_lambda_values,
        dataset_name="rna_ai",
        top_n_genes=None,
        gamma=0.01,
        eta=0.5,
        max_iter=500,
        output_dir=output_dir,
    ) # Results imply we should use lambda=0.136 for rna 