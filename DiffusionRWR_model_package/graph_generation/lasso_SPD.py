import numpy as np
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso

from ..data_sorting import load_data, std_data_dict


def test_adjacency_symmetry(adjacency, atol=1e-8, rtol=1e-5):
    """
    Test whether an adjacency matrix is symmetric within numeric tolerances.

    Parameters
    ----------
    adjacency : np.ndarray or pd.DataFrame
        Square adjacency matrix.
    atol : float
        Absolute tolerance for symmetry check.
    rtol : float
        Relative tolerance for symmetry check.

    Returns
    -------
    dict
        {
            "is_symmetric": bool,
            "max_abs_asymmetry": float,
            "shape": tuple[int, int]
        }
    """
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=False)
    else:
        matrix = np.asarray(adjacency, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency must be a square 2D matrix, got shape {matrix.shape}")

    asymmetry = matrix - matrix.T
    max_abs_asymmetry = float(np.max(np.abs(asymmetry)))
    is_symmetric = bool(np.allclose(matrix, matrix.T, atol=atol, rtol=rtol, equal_nan=True))

    return {
        "is_symmetric": is_symmetric,
        "max_abs_asymmetry": max_abs_asymmetry,
        "shape": tuple(matrix.shape),
    }


def test_adjacency_positive_semidefinite(adjacency, atol=1e-8):
    """
    Test whether an adjacency matrix is positive semi-definite.

    Parameters
    ----------
    adjacency : np.ndarray or pd.DataFrame
        Square adjacency matrix.
    atol : float
        Absolute tolerance for eigenvalue check (minimum eigenvalue >= -atol).

    Returns
    -------
    dict
        {
            "is_positive_semidefinite": bool,
            "min_eigenvalue": float,
            "max_eigenvalue": float,
            "condition_number": float,
            "shape": tuple[int, int]
        }
    """
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=False)
    else:
        matrix = np.asarray(adjacency, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency must be a square 2D matrix, got shape {matrix.shape}")

    # Ensure symmetry for eigenvalue computation
    matrix = (matrix + matrix.T) / 2

    eigvals = np.linalg.eigvals(matrix)
    eigvals = np.real(eigvals)  # Take real part in case of numerical issues

    min_eigenvalue = float(np.min(eigvals))
    max_eigenvalue = float(np.max(eigvals))
    is_positive_semidefinite = bool(min_eigenvalue >= -atol)

    # Condition number (largest / smallest positive eigenvalue)
    positive_eigvals = eigvals[eigvals > atol]
    if positive_eigvals.size > 0:
        condition_number = float(np.max(positive_eigvals) / np.min(positive_eigvals))
    else:
        condition_number = np.inf

    return {
        "is_positive_semidefinite": is_positive_semidefinite,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "condition_number": condition_number,
        "shape": tuple(matrix.shape),
    }


def get_adjacency_element_range(adjacency):
    """
    Get the range (min, max) of elements in the adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray or pd.DataFrame
        Adjacency matrix.

    Returns
    -------
    dict
        {
            "min_value": float,
            "max_value": float,
            "range": float,
            "has_nan": bool,
            "has_inf": bool
        }
    """
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=False)
    else:
        matrix = np.asarray(adjacency, dtype=float)

    min_value = float(np.nanmin(matrix))
    max_value = float(np.nanmax(matrix))
    range_value = max_value - min_value
    has_nan = bool(np.any(np.isnan(matrix)))
    has_inf = bool(np.any(np.isinf(matrix)))

    return {
        "min_value": min_value,
        "max_value": max_value,
        "range": range_value,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }

def f(omega, X, tau):
    """Regression loss"""
    p = X.shape[1]
    b = []
    for j in range(p):
        idx = np.arange(p) != j
        r = X[:, j] + tau[j] * X[:, idx] @ omega[idx, j]
        b.append(np.linalg.norm(r)**2)
    return np.sum(b)


def g(omega, tau, lam):
    """L1 penalty"""
    p = omega.shape[0]
    s = 0
    for j in range(p):
        idx = np.arange(p) != j
        s += lam * tau[j] * np.sum(np.abs(omega[idx, j]))
    return s


def PI(omega, alpha=1e-6):
    """Projection onto SPD cone"""
    omega = (omega + omega.T) / 2
    eigvals, eigvecs = np.linalg.eigh(omega)
    eigvals_clipped = np.clip(eigvals, a_min=alpha, a_max=None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def gradient_f(omega, X, tau):
    """Analytical gradient of f with tau treated as fixed within each proximal step."""
    p = omega.shape[0]
    grad = np.zeros_like(omega)

    for j in range(p):
        idx = np.arange(p) != j
        A = X[:, idx]
        w = omega[idx, j]
        r = X[:, j] + tau[j] * (A @ w)
        grad[idx, j] = 2.0 * tau[j] * (A.T @ r)

    np.fill_diagonal(grad, 0.0)

    return grad


def next_omega(omega, gamma, U, grad_f):
    """Primal update"""
    omega_next = omega - gamma * U - gamma * grad_f
    return PI(omega_next)


import numpy as np

def prox_g_over_eta(Z, tau_squared, lam, eta):
    """
    Proximal operator of g/eta applied elementwise.

    Parameters:
        Z : (p, p) matrix  (this is V / eta)
        tau_hat_sq : (p,) vector of τ̂_j^2
        lam : (p,) or scalar λ_j
        eta : scalar

    Returns:
        prox matrix of same shape
    """
    p = Z.shape[0]
    prox = np.zeros_like(Z)

    for j in range(p):
        # Diagonal
        prox[j, j] = 1.0 / tau_squared[j]

        # Off-diagonal
        for k in range(p):
            if k == j:
                continue
            
            threshold = tau_squared[j] * lam / eta
            prox[k, j] = np.sign(Z[k, j]) * max(abs(Z[k, j]) - threshold, 0.0)

    return prox


def dual_update(U_k, Omega_k, Omega_k1, grad_k, grad_k1, eta, gamma, tau_squared, lam):
    """
    Perform the dual update step.

    Parameters:
        U_k        : current dual variable (p x p)
        Omega_k    : Ω^{(k)}
        Omega_k1   : Ω^{(k+1)}
        grad_k     : ∇f(Ω^{(k)})
        grad_k1    : ∇f(Ω^{(k+1)})
        eta, gamma : step sizes
        tau_squared : vector (p,)
        lam        : scalar or vector (p,)

    Returns:
        U_{k+1}
    """

    # Step 1: compute V^{(k+1)}
    V = (
        U_k
        + eta * (
            2 * Omega_k1 - Omega_k
            + gamma * (grad_k - grad_k1)
        )
    )

    # Step 2: proximal step via Moreau identity
    Z = V / eta
    prox = prox_g_over_eta(Z, tau_squared, lam, eta)

    U_k1 = V - eta * prox

    return U_k1


def initialize_omega_lasso(X, lam=0.1, max_iter_inner=100):
    """
    Column-wise Lasso regression using sklearn.
    Solves: min_w (1/2n)||X_j - X_{-j} w||^2 + lam ||w||_1
    """
    n, p = X.shape
    omega = np.eye(p)
    thetas= np.zeros((p, p))
    tau_squared = np.zeros(p)
    
    for j in range(p):
        idx = np.arange(p) != j
        
        X_j = X[:, j]
        X_rest = X[:, idx]
        
        # sklearn uses (1/2n)||y - Xw||^2 + alpha ||w||_1
        model = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter_inner)
        model.fit(X_rest, X_j)
        
        thetas[idx, j] = model.coef_

        # Compute residual variance for tau_j
        residuals = X_j - X_rest @ model.coef_
        tau_squared[j] = np.mean(residuals**2)

        omega[idx, j] = -thetas[idx, j] / max(tau_squared[j], 1e-6)  # Avoid division by zero
        omega[j,j] = 1.0 / max(tau_squared[j], 1e-6)  # Diagonal entry for precision
    
    return omega, tau_squared


def proximal_precision(X, lam=0.1, gamma=0.01, eta=0.5, max_iter=100, return_omega=False, show_ls_loss=False):
    """
    Proximal splitting solver with dual variables for SPD-Lasso estimation.
    
    Solves: min_omega f(omega) + g(omega) subject to omega in SPD_cone
    where f(omega) = regression loss, g(omega) = lambda * L1 penalty
    
    Uses alternating primal-dual updates with eta as dual step parameter.
    """
    _, p = X.shape

    # Initialize omega using lasso regression
    omega, tau_squared = initialize_omega_lasso(X, lam=lam, max_iter_inner=10000)
    
    # Initialize dual variable
    U = np.zeros((p, p))
    
    losses = []
    ls_losses = []
    print("starting proximal solver...")
    for iteration in range(max_iter):
        grad_old = gradient_f(omega, X, tau_squared)
        
        # Primal update
        omega_old = omega.copy()
        omega = next_omega(omega, gamma, U, grad_old)
        omega = PI(omega, alpha=1e-6)
        
        grad_new = gradient_f(omega, X, tau_squared)
        
        # Dual update (using eta)
        U = dual_update(U, omega_old, omega, grad_old, grad_new, eta, gamma, tau_squared, lam)
        
        # Compute loss
        ls_loss = f(omega, X, tau_squared)
        l1_loss = g(omega, tau_squared, lam)
        total_loss = ls_loss + l1_loss
        
        losses.append(total_loss)
        ls_losses.append(ls_loss)
        
        if show_ls_loss:
            print(f"Iter {iteration + 1}: LS loss: {ls_loss:.6f}, L1 loss: {l1_loss:.6f}, Total: {total_loss:.6f}")
        if iteration > 0 and abs(losses[-2] - losses[-1]) < 1e-6:
            print(f"Convergence reached at iteration {iteration + 1}.")
            break
    if return_omega:
        return omega, losses
    else:
        T = np.diag(np.sqrt(tau_squared))
        Q = -T @ omega @ T
        return Q, losses


def run_proximal_on_rna_data(
    folder_path,
    rna_key_contains="rna",
    top_n_genes=60,
    lam=0.1,
    gamma=0.01,
    eta=0.5,
    max_iter=100,
    make_plot=True,
):
    """
    Load an RNA dataset from `folder_path`, run the proximal splitting solver,
    and optionally plot objective loss over iterations.

    Returns
    -------
    omega : np.ndarray
        Estimated SPD precision matrix.
    losses : list[float]
        Objective values per iteration.
    metadata : dict
        Dataset/solver metadata for reproducibility.
    """
    data_dict = load_data(folder_path)
    match_token = str(rna_key_contains).lower()

    rna_candidates = [
        key for key in data_dict.keys()
        if match_token in key.lower()
    ]
    if not rna_candidates:
        raise ValueError(
            f"No RNA dataset found using token '{rna_key_contains}' in folder '{folder_path}'."
        )

    selected_key = sorted(rna_candidates)[0]
    rna_df = data_dict[selected_key].copy()
    rna_df = rna_df.dropna(axis=0, how="any")

    if top_n_genes is not None and int(top_n_genes) > 0 and len(rna_df) > int(top_n_genes):
        top_idx = rna_df.var(axis=1).sort_values(ascending=False).head(int(top_n_genes)).index
        rna_df = rna_df.loc[top_idx]

    if rna_df.empty:
        raise ValueError("RNA dataset is empty after preprocessing/filtering.")

    X = rna_df.T.to_numpy(dtype=float)
    Q, losses = proximal_precision(
        X,
        lam=lam,
        gamma=gamma,
        eta=eta,
        max_iter=max_iter,
    )

    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=2)
        ax.set_title(f"Proximal SPD Lasso Loss ({selected_key})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    metadata = {
        "folder_path": str(Path(folder_path)),
        "selected_dataset": selected_key,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "top_n_genes": int(top_n_genes) if top_n_genes is not None else None,
        "lam": float(lam),
        "gamma": float(gamma),
        "eta": float(eta),
        "max_iter": int(max_iter),
    }
    return Q, losses, metadata


def run_proximal_hyperparameter_sweep(
    folder_path,
    rna_key_contains="rna",
    lam_values=(0.01, 0.05, 0.1),
    gamma_values=(0.001, 0.005, 0.01),
    eta_values=(0.25, 0.5),
    max_iter_values=(100,),
    top_n_genes_values=(60,),
    sort_by="final_loss",
    ascending=True,
    make_best_plot=True,
):
    """
    Run proximal SPD-Lasso over a hyperparameter grid.

    Returns
    -------
    results_df : pd.DataFrame
        One row per hyperparameter combination with convergence metrics.
    best_run : dict | None
        Best run payload with keys: Q, losses, metadata, row.
    """
    rows = []
    best_run = None

    for lam, gamma, eta, max_iter, top_n_genes in product(
        lam_values,
        gamma_values,
        eta_values,
        max_iter_values,
        top_n_genes_values,
    ):
        try:
            Q, losses, metadata = run_proximal_on_rna_data(
                folder_path=folder_path,
                rna_key_contains=rna_key_contains,
                top_n_genes=top_n_genes,
                lam=lam,
                gamma=gamma,
                eta=eta,
                max_iter=max_iter,
                make_plot=False,
            )

            loss_arr = np.asarray(losses, dtype=float)
            first_loss = float(loss_arr[0]) if loss_arr.size else np.nan
            final_loss = float(loss_arr[-1]) if loss_arr.size else np.nan
            min_idx = int(np.argmin(loss_arr)) if loss_arr.size else -1
            min_loss = float(loss_arr[min_idx]) if loss_arr.size else np.nan
            best_iter = int(min_idx + 1) if loss_arr.size else np.nan
            monotone_nonincreasing = bool(np.all(np.diff(loss_arr) <= 1e-12)) if loss_arr.size > 1 else True
            delta_loss = final_loss - first_loss if loss_arr.size else np.nan

            row = {
                "status": "ok",
                "lam": float(lam),
                "gamma": float(gamma),
                "eta": float(eta),
                "max_iter": int(max_iter),
                "top_n_genes": int(top_n_genes),
                "n_losses": int(loss_arr.size),
                "first_loss": first_loss,
                "final_loss": final_loss,
                "min_loss": min_loss,
                "best_iter": best_iter,
                "delta_loss": float(delta_loss),
                "monotone_nonincreasing": monotone_nonincreasing,
                "selected_dataset": metadata.get("selected_dataset"),
                "n_samples": metadata.get("n_samples"),
                "n_features": metadata.get("n_features"),
                "error": "",
            }
            rows.append(row)

            if best_run is None:
                best_run = {
                    "Q": Q,
                    "losses": losses,
                    "metadata": metadata,
                    "row": row,
                }
            else:
                current_best = best_run["row"].get(sort_by, np.inf)
                candidate = row.get(sort_by, np.inf)
                if (ascending and candidate < current_best) or (not ascending and candidate > current_best):
                    best_run = {
                        "Q": Q,
                        "losses": losses,
                        "metadata": metadata,
                        "row": row,
                    }

        except Exception as exc:
            rows.append(
                {
                    "status": "failed",
                    "lam": float(lam),
                    "gamma": float(gamma),
                    "eta": float(eta),
                    "max_iter": int(max_iter),
                    "top_n_genes": int(top_n_genes),
                    "n_losses": 0,
                    "first_loss": np.nan,
                    "final_loss": np.nan,
                    "min_loss": np.nan,
                    "best_iter": np.nan,
                    "delta_loss": np.nan,
                    "monotone_nonincreasing": False,
                    "selected_dataset": "",
                    "n_samples": np.nan,
                    "n_features": np.nan,
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(rows)
    if not results_df.empty and sort_by in results_df.columns:
        results_df = results_df.sort_values(by=["status", sort_by], ascending=[True, ascending]).reset_index(drop=True)

    if make_best_plot and best_run is not None:
        best_losses = np.asarray(best_run["losses"], dtype=float)
        if best_losses.size:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(np.arange(1, best_losses.size + 1), best_losses, linewidth=2)
            best_label = (
                f"Best Sweep Loss | lam={best_run['row']['lam']}, "
                f"gamma={best_run['row']['gamma']}, eta={best_run['row']['eta']}"
            )
            ax.set_title(best_label)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Objective loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

    return results_df, best_run


def run_spd_lasso_symmetry_check_on_rna_ai(
    top_n_genes=60,
    lam=0.1,
    gamma=0.01,
    eta=0.5,
    max_iter=100,
    atol=1e-8,
    rtol=1e-5,
):
    if "rna_ai" not in std_data_dict:
        raise KeyError("'rna_ai' not found in std_data_dict")

    rna_df = std_data_dict["rna_ai"].copy().dropna(axis=0, how="any")

    if top_n_genes is not None and int(top_n_genes) > 0 and len(rna_df) > int(top_n_genes):
        top_idx = rna_df.var(axis=1).sort_values(ascending=False).head(int(top_n_genes)).index
        rna_df = rna_df.loc[top_idx]

    if rna_df.empty:
        raise ValueError("rna_ai dataset is empty after preprocessing/filtering")

    X = rna_df.T.to_numpy(dtype=float)
    adjacency, losses = proximal_precision(
        X,
        lam=lam,
        gamma=gamma,
        eta=eta,
        max_iter=max_iter,
    )

    symmetry_report = test_adjacency_symmetry(adjacency, atol=atol, rtol=rtol)

    report = {
        "dataset": "rna_ai",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "top_n_genes": int(top_n_genes) if top_n_genes is not None else None,
        "lam": float(lam),
        "gamma": float(gamma),
        "eta": float(eta),
        "max_iter": int(max_iter),
        "final_loss": float(losses[-1]) if losses else np.nan,
        **symmetry_report,
    }

    return adjacency, losses, report


def main():
    lam = 0.5
    
    # Test Q (adjacency matrix)
    adjacency, losses, report = run_spd_lasso_symmetry_check_on_rna_ai(
        top_n_genes=None,
        lam=lam,
        gamma=0.01,
        eta=0.5,
        max_iter=100,
    )

    # Test omega (precision matrix)
    omega, losses_omega = proximal_precision(
        std_data_dict["rna_ai"].T.to_numpy(dtype=float),
        lam=lam,
        gamma=0.01,
        eta=0.5,
        max_iter=1000,
        return_omega=True,
        show_ls_loss=True,
    )

    # Get X for computing loss components
    X = std_data_dict["rna_ai"].T.to_numpy(dtype=float)
    tau = 1 / np.diag(omega)
    
    # Compute loss components for omega
    f_value = f(omega, X, tau)
    g_value = g(omega, tau, lam)
    total_loss = f_value + g_value

    # Additional tests for Q
    psd_report_q = test_adjacency_positive_semidefinite(adjacency)
    range_report_q = get_adjacency_element_range(adjacency)

    # Additional tests for omega
    psd_report_omega = test_adjacency_positive_semidefinite(omega)
    range_report_omega = get_adjacency_element_range(omega)

    print("\nSPD-Lasso tests on std_data_dict['rna_ai']")
    print("=" * 50)
    print("ADJACENCY MATRIX (Q = -T @ omega @ T):")
    print(f"Shape: {adjacency.shape}")
    print(f"Iterations: {len(losses)}")
    print(f"Final loss: {report['final_loss']:.6f}")
    print()
    print("Symmetry check:")
    print(f"  Symmetric: {report['is_symmetric']}")
    print(f"  Max |A - A^T|: {report['max_abs_asymmetry']:.6e}")
    print()
    print("Positive semi-definite check:")
    print(f"  PSD: {psd_report_q['is_positive_semidefinite']}")
    print(f"  Min eigenvalue: {psd_report_q['min_eigenvalue']:.6e}")
    print(f"  Max eigenvalue: {psd_report_q['max_eigenvalue']:.6e}")
    print(f"  Condition number: {psd_report_q['condition_number']:.6e}")
    print()
    print("Element range:")
    print(f"  Min value: {range_report_q['min_value']:.6e}")
    print(f"  Max value: {range_report_q['max_value']:.6e}")
    print(f"  Range: {range_report_q['range']:.6e}")
    print(f"  Has NaN: {range_report_q['has_nan']}")
    print(f"  Has Inf: {range_report_q['has_inf']}")
    print()
    print("=" * 50)
    print("PRECISION MATRIX (omega):")
    print(f"Shape: {omega.shape}")
    print(f"Iterations: {len(losses_omega)}")
    print(f"Final loss: {losses_omega[-1]:.6f}")
    print()
    print("Loss components (lam={}):".format(lam))
    print(f"  Least squares loss f(omega): {f_value:.6f}")
    print(f"  L1 penalty g(omega): {g_value:.6f}")
    print(f"  Total loss: {total_loss:.6f}")
    print()
    print("Positive semi-definite check:")
    print(f"  PSD: {psd_report_omega['is_positive_semidefinite']}")
    print(f"  Min eigenvalue: {psd_report_omega['min_eigenvalue']:.6e}")
    print(f"  Max eigenvalue: {psd_report_omega['max_eigenvalue']:.6e}")
    print(f"  Condition number: {psd_report_omega['condition_number']:.6e}")
    print()
    print("Element range:")
    print(f"  Min value: {range_report_omega['min_value']:.6e}")
    print(f"  Max value: {range_report_omega['max_value']:.6e}")
    print(f"  Range: {range_report_omega['range']:.6e}")
    print(f"  Has NaN: {range_report_omega['has_nan']}")
    print(f"  Has Inf: {range_report_omega['has_inf']}")


if __name__ == "__main__":
    main()

