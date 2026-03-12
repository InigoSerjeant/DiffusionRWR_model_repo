import numpy as np
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

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


def update_tau(omega):
    """tau_j = 1 / omega_jj"""
    return 1 / np.diag(omega)


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


def _soft_threshold_offdiag(omega, step_size, tau, lam):
    """Column-wise weighted soft-thresholding on off-diagonal entries."""
    out = omega.copy()
    p = omega.shape[0]
    for j in range(p):
        idx = np.arange(p) != j
        threshold = step_size * lam * tau[j]
        values = out[idx, j]
        out[idx, j] = np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
    out = 0.5 * (out + out.T)
    return out


def prox_g_star(U, tau, lam):
    p = len(tau)
    U_new = U.copy()
    for j in range(p):
        U_new[:, j] = np.clip(U[:, j], -lam * tau[j], lam * tau[j])
    return U_new


def next_U(U, omega_new, omega_old, grad_old, grad_new, tau, eta, gamma, lam):
    """Dual update"""
    Z = (
        U
        + eta * (2 * omega_new - omega_old)
        + gamma * eta * (grad_old - grad_new)
    )
    return prox_g_star(Z, tau, lam)


def proximal_precision(X, lam=0.1, gamma=0.01, eta=0.5, max_iter=100):
    """
    Main proximal splitting solver
    """
    _, p = X.shape

    omega = np.eye(p)

    losses = []
    step_size = float(gamma)

    for _ in range(max_iter):
        tau = update_tau(omega)
        grad = gradient_f(omega, X, tau)
        current_loss_fixed_tau = f(omega, X, tau) + g(omega, tau, lam)

        local_step = step_size
        omega_candidate = omega

        for _ in range(20):
            trial = omega - local_step * grad
            trial = _soft_threshold_offdiag(trial, local_step, tau, lam)
            trial = PI(trial)

            trial_loss_fixed_tau = f(trial, X, tau) + g(trial, tau, lam)

            if np.isfinite(trial_loss_fixed_tau) and trial_loss_fixed_tau <= current_loss_fixed_tau:
                omega_candidate = trial
                break

            local_step *= 0.5

        omega = omega_candidate
        step_size = min(max(local_step * 1.05, 1e-8), max(float(gamma), 1e-8))

        tau_eval = update_tau(omega)
        losses.append(f(omega, X, tau_eval) + g(omega, tau_eval, lam))
    T = np.diag(np.sqrt(update_tau(omega)))
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
    adjacency, losses, report = run_spd_lasso_symmetry_check_on_rna_ai(
        top_n_genes=None,
        lam=0.001,
        gamma=0.01,
        eta=0.5,
        max_iter=100,
    )

    print("\nSPD-Lasso symmetry check on std_data_dict['rna_ai']")
    print(f"Adjacency shape: {adjacency.shape}")
    print(f"Iterations: {len(losses)}")
    print(f"Final loss: {report['final_loss']:.6f}")
    print(f"Symmetric: {report['is_symmetric']}")
    print(f"Max |A - A^T|: {report['max_abs_asymmetry']:.6e}")


if __name__ == "__main__":
    main()

