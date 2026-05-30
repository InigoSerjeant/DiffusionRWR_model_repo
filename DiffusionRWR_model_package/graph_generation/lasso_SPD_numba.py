import numpy as np
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, prange
from sklearn.linear_model import Lasso

from ..data_sorting import load_data, std_data_dict


MAX_ABS_MATRIX_VALUE = 1e6
MIN_DIAG_VALUE = 1e-6


@njit(cache=True)
def _sanitize_and_clip_matrix_numba(A, clip_value):
    out = np.empty_like(A)
    n, m = A.shape
    for i in range(n):
        for j in range(m):
            v = A[i, j]
            if np.isnan(v):
                v = 0.0
            elif np.isinf(v):
                if v > 0.0:
                    v = clip_value
                else:
                    v = -clip_value
            elif v > clip_value:
                v = clip_value
            elif v < -clip_value:
                v = -clip_value
            out[i, j] = v
    return out


@njit(cache=True)
def _sanitize_tau_numba(tau, min_value, max_value):
    out = np.empty_like(tau)
    for i in range(tau.shape[0]):
        v = tau[i]
        if np.isnan(v) or np.isinf(v):
            v = min_value
        if v < min_value:
            v = min_value
        elif v > max_value:
            v = max_value
        out[i] = v
    return out


@njit(cache=True)
def _f_numba(omega, X, tau):
    p = X.shape[1]
    total = 0.0
    for j in range(p):
        b = 0.0
        for i in range(X.shape[0]):
            acc = X[i, j]
            for k in range(p):
                if k != j:
                    acc += tau[j] * X[i, k] * omega[k, j]
            b += acc * acc
        total += b
    return total


@njit(cache=True, parallel=True)
def _f_numba_from_gram(omega, gram, tau):
    p = omega.shape[0]
    col_totals = np.empty(p, dtype=np.float64)

    for j in prange(p):
        tau_j = tau[j]

        # term1 = x_j^T x_j
        term1 = gram[j, j]

        # term2 = 2 * tau_j * w^T X_-j^T x_j
        term2_inner = 0.0
        for k in range(j):
            term2_inner += omega[k, j] * gram[k, j]
        for k in range(j + 1, p):
            term2_inner += omega[k, j] * gram[k, j]
        term2 = 2.0 * tau_j * term2_inner

        # term3 = tau_j^2 * w^T X_-j^T X_-j w
        quad = 0.0
        for k in range(j):
            wk = omega[k, j]
            row_acc = 0.0
            for m in range(j):
                row_acc += gram[k, m] * omega[m, j]
            for m in range(j + 1, p):
                row_acc += gram[k, m] * omega[m, j]
            quad += wk * row_acc
        for k in range(j + 1, p):
            wk = omega[k, j]
            row_acc = 0.0
            for m in range(j):
                row_acc += gram[k, m] * omega[m, j]
            for m in range(j + 1, p):
                row_acc += gram[k, m] * omega[m, j]
            quad += wk * row_acc
        term3 = (tau_j * tau_j) * quad

        col_totals[j] = term1 + term2 + term3

    total = 0.0
    for j in range(p):
        total += col_totals[j]

    return total


@njit(cache=True, parallel=True)
def _g_numba(omega, tau, lam):
    p = omega.shape[0]
    col_penalty = np.empty(p, dtype=np.float64)
    for j in prange(p):
        col_sum = 0.0
        for k in range(j):
            col_sum += abs(omega[k, j])
        for k in range(j + 1, p):
            col_sum += abs(omega[k, j])
        col_penalty[j] = lam * tau[j] * col_sum

    s = 0.0
    for j in range(p):
        s += col_penalty[j]
    return s


@njit(cache=True)
def _pi_numba(omega, alpha=1e-6):
    omega = _sanitize_and_clip_matrix_numba(omega, MAX_ABS_MATRIX_VALUE)
    omega_sym = (omega + omega.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(omega_sym)

    p = eigvals.shape[0]
    eigvals_clipped = np.empty_like(eigvals)
    for i in range(p):
        if eigvals[i] < alpha:
            eigvals_clipped[i] = alpha
        else:
            eigvals_clipped[i] = eigvals[i]

    # Equivalent to eigvecs @ diag(eigvals_clipped) but avoids building an explicit dense diagonal matrix.
    scaled_eigvecs = eigvecs * eigvals_clipped
    return scaled_eigvecs @ eigvecs.T


@njit(cache=True)
def _gradient_f_numba(omega, X, tau):
    p = omega.shape[0]
    n = X.shape[0]
    grad = np.zeros_like(omega)

    for j in range(p):
        for k in range(p):
            if k == j:
                continue

            acc = 0.0
            for i in range(n):
                r_i = X[i, j]
                for m in range(p):
                    if m != j:
                        r_i += tau[j] * X[i, m] * omega[m, j]
                acc += X[i, k] * r_i

            grad[k, j] = 2.0 * tau[j] * acc

    for j in range(p):
        grad[j, j] = 0.0

    return grad


@njit(cache=True, parallel=True)
def _gradient_f_numba_from_gram_inplace(omega, gram, tau, grad):
    p = omega.shape[0]

    for j in prange(p):
        tau_j = tau[j]
        grad[j, j] = 0.0
        for k in range(j):
            acc = gram[k, j]
            weighted = 0.0
            for m in range(j):
                weighted += gram[k, m] * omega[m, j]
            for m in range(j + 1, p):
                weighted += gram[k, m] * omega[m, j]

            acc += tau_j * weighted
            grad[k, j] = 2.0 * tau_j * acc

        for k in range(j + 1, p):
            acc = gram[k, j]
            weighted = 0.0
            for m in range(j):
                weighted += gram[k, m] * omega[m, j]
            for m in range(j + 1, p):
                weighted += gram[k, m] * omega[m, j]

            acc += tau_j * weighted
            grad[k, j] = 2.0 * tau_j * acc


@njit(cache=True)
def _gradient_f_numba_from_gram(omega, gram, tau):
    grad = np.zeros_like(omega)
    _gradient_f_numba_from_gram_inplace(omega, gram, tau, grad)

    return grad


@njit(cache=True)
def _next_omega_numba(omega, gamma, U, grad_f):
    omega_next = omega - gamma * U - gamma * grad_f
    omega_next = _sanitize_and_clip_matrix_numba(omega_next, MAX_ABS_MATRIX_VALUE)
    return _pi_numba(omega_next)


@njit(cache=True)
def _prox_g_over_eta_numba(Z, tau_squared, lam, eta):
    p = Z.shape[0]
    prox = np.zeros_like(Z)

    for j in range(p):
        prox[j, j] = 1.0 / tau_squared[j]
        threshold = tau_squared[j] * lam / eta

        for k in range(p):
            if k == j:
                continue

            z = Z[k, j]
            az = abs(z)
            shrunk = az - threshold
            if shrunk < 0.0:
                shrunk = 0.0

            if z > 0.0:
                prox[k, j] = shrunk
            elif z < 0.0:
                prox[k, j] = -shrunk
            else:
                prox[k, j] = 0.0

    return prox


@njit(cache=True)
def _dual_update_numba(U_k, Omega_k, Omega_k1, grad_k, grad_k1, eta, gamma, tau_squared, lam):
    V = U_k + eta * (2.0 * Omega_k1 - Omega_k + gamma * (grad_k - grad_k1))
    Z = V / eta
    prox = _prox_g_over_eta_numba(Z, tau_squared, lam, eta)
    return V - eta * prox


@njit(cache=True)
def _proximal_precision_core_numba(omega, U, X, tau_squared, lam, gamma, eta, max_iter, tol):
    X = _sanitize_and_clip_matrix_numba(X, MAX_ABS_MATRIX_VALUE)
    omega = _sanitize_and_clip_matrix_numba(omega, MAX_ABS_MATRIX_VALUE)
    U = _sanitize_and_clip_matrix_numba(U, MAX_ABS_MATRIX_VALUE)
    tau_squared = _sanitize_tau_numba(tau_squared, MIN_DIAG_VALUE, MAX_ABS_MATRIX_VALUE)

    gram = X.T @ X
    gram = _sanitize_and_clip_matrix_numba(gram, MAX_ABS_MATRIX_VALUE)
    losses = np.empty(max_iter, dtype=np.float64)
    ls_losses = np.empty(max_iter, dtype=np.float64)
    omega_old = np.empty_like(omega)
    grad_old = np.empty_like(omega)
    grad_new = np.empty_like(omega)

    n_iters = 0
    converged = 0

    for iteration in range(max_iter):
        _gradient_f_numba_from_gram_inplace(omega, gram, tau_squared, grad_old)
        grad_old = _sanitize_and_clip_matrix_numba(grad_old, MAX_ABS_MATRIX_VALUE)

        omega_old[:, :] = omega
        omega = _next_omega_numba(omega, gamma, U, grad_old)
        # Removed redundant PSD projection
        # omega = _pi_numba(omega, alpha=1e-6)

        _gradient_f_numba_from_gram_inplace(omega, gram, tau_squared, grad_new)
        grad_new = _sanitize_and_clip_matrix_numba(grad_new, MAX_ABS_MATRIX_VALUE)
        U = _dual_update_numba(U, omega_old, omega, grad_old, grad_new, eta, gamma, tau_squared, lam)
        U = _sanitize_and_clip_matrix_numba(U, MAX_ABS_MATRIX_VALUE)

        ls_loss = _f_numba_from_gram(omega, gram, tau_squared)
        l1_loss = _g_numba(omega, tau_squared, lam)
        total_loss = ls_loss + l1_loss

        losses[iteration] = total_loss
        ls_losses[iteration] = ls_loss
        n_iters = iteration + 1

        if iteration > 0 and abs(losses[iteration - 1] - losses[iteration]) < tol:
            converged = 1
            break

    return omega, U, losses[:n_iters], ls_losses[:n_iters], converged, n_iters


def test_adjacency_symmetry(adjacency, atol=1e-8, rtol=1e-5):
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
    if isinstance(adjacency, pd.DataFrame):
        matrix = adjacency.to_numpy(dtype=float, copy=False)
    else:
        matrix = np.asarray(adjacency, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency must be a square 2D matrix, got shape {matrix.shape}")

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=MAX_ABS_MATRIX_VALUE, neginf=-MAX_ABS_MATRIX_VALUE)
    matrix = np.clip(matrix, -MAX_ABS_MATRIX_VALUE, MAX_ABS_MATRIX_VALUE)
    matrix = (matrix + matrix.T) / 2

    eigvals = np.linalg.eigvals(matrix)
    eigvals = np.real(eigvals)

    min_eigenvalue = float(np.min(eigvals))
    max_eigenvalue = float(np.max(eigvals))
    is_positive_semidefinite = bool(min_eigenvalue >= -atol)

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
    return _f_numba(np.asarray(omega, dtype=np.float64), np.asarray(X, dtype=np.float64), np.asarray(tau, dtype=np.float64))


def g(omega, tau, lam):
    return _g_numba(np.asarray(omega, dtype=np.float64), np.asarray(tau, dtype=np.float64), float(lam))


def PI(omega, alpha=1e-6):
    return _pi_numba(np.asarray(omega, dtype=np.float64), float(alpha))


def gradient_f(omega, X, tau):
    return _gradient_f_numba(np.asarray(omega, dtype=np.float64), np.asarray(X, dtype=np.float64), np.asarray(tau, dtype=np.float64))


def next_omega(omega, gamma, U, grad_f):
    return _next_omega_numba(
        np.asarray(omega, dtype=np.float64),
        float(gamma),
        np.asarray(U, dtype=np.float64),
        np.asarray(grad_f, dtype=np.float64),
    )


def prox_g_over_eta(Z, tau_squared, lam, eta):
    return _prox_g_over_eta_numba(
        np.asarray(Z, dtype=np.float64),
        np.asarray(tau_squared, dtype=np.float64),
        float(lam),
        float(eta),
    )


def dual_update(U_k, Omega_k, Omega_k1, grad_k, grad_k1, eta, gamma, tau_squared, lam):
    return _dual_update_numba(
        np.asarray(U_k, dtype=np.float64),
        np.asarray(Omega_k, dtype=np.float64),
        np.asarray(Omega_k1, dtype=np.float64),
        np.asarray(grad_k, dtype=np.float64),
        np.asarray(grad_k1, dtype=np.float64),
        float(eta),
        float(gamma),
        np.asarray(tau_squared, dtype=np.float64),
        float(lam),
    )


def initialize_omega_lasso(X, lam=0.1, max_iter_inner=100):
    n, p = X.shape
    omega = np.eye(p)
    thetas = np.zeros((p, p))
    tau_squared = np.zeros(p)

    for j in range(p):
        idx = np.arange(p) != j

        X_j = X[:, j]
        X_rest = X[:, idx]

        model = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter_inner)
        model.fit(X_rest, X_j)

        thetas[idx, j] = model.coef_

        residuals = X_j - X_rest @ model.coef_
        tau_squared[j] = np.mean(residuals ** 2)

        omega[idx, j] = -thetas[idx, j] / max(tau_squared[j], 1e-6)
        omega[j, j] = 1.0 / max(tau_squared[j], 1e-6)

    return omega, tau_squared


def proximal_precision(X, lam=0.1, gamma=0.01, eta=0.5, max_iter=100, return_omega=False, show_ls_loss=False):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    X = np.nan_to_num(X, nan=0.0, posinf=MAX_ABS_MATRIX_VALUE, neginf=-MAX_ABS_MATRIX_VALUE)
    X = np.clip(X, -MAX_ABS_MATRIX_VALUE, MAX_ABS_MATRIX_VALUE)
    _, p = X.shape

    omega, tau_squared = initialize_omega_lasso(X, lam=lam, max_iter_inner=10000)
    omega = np.ascontiguousarray(np.asarray(omega, dtype=np.float64))
    tau_squared = np.ascontiguousarray(np.asarray(tau_squared, dtype=np.float64))
    tau_squared = np.nan_to_num(tau_squared, nan=MIN_DIAG_VALUE, posinf=MAX_ABS_MATRIX_VALUE, neginf=MIN_DIAG_VALUE)
    tau_squared = np.clip(tau_squared, MIN_DIAG_VALUE, MAX_ABS_MATRIX_VALUE)
    U = np.zeros((p, p), dtype=np.float64)

    print("starting proximal solver...")
    omega, U, losses, ls_losses, converged, n_iters = _proximal_precision_core_numba(
        omega,
        U,
        X,
        tau_squared,
        float(lam),
        float(gamma),
        float(eta),
        int(max_iter),
        1e-6,
    )

    if show_ls_loss:
        for iteration in range(n_iters):
            ls_loss = float(ls_losses[iteration])
            total_loss = float(losses[iteration])
            l1_loss = total_loss - ls_loss
            print(f"Iter {iteration + 1}: LS loss: {ls_loss:.6f}, L1 loss: {l1_loss:.6f}, Total: {total_loss:.6f}")

    if converged == 1:
        print(f"Convergence reached at iteration {n_iters}.")

    if return_omega:
        omega = np.nan_to_num(omega, nan=0.0, posinf=MAX_ABS_MATRIX_VALUE, neginf=-MAX_ABS_MATRIX_VALUE)
        omega = np.clip(omega, -MAX_ABS_MATRIX_VALUE, MAX_ABS_MATRIX_VALUE)
        return omega, losses.tolist()

    T = np.diag(np.sqrt(tau_squared))
    Q = -T @ omega @ T
    Q = np.nan_to_num(Q, nan=0.0, posinf=MAX_ABS_MATRIX_VALUE, neginf=-MAX_ABS_MATRIX_VALUE)
    Q = np.clip(Q, -MAX_ABS_MATRIX_VALUE, MAX_ABS_MATRIX_VALUE)
    return Q, losses.tolist()


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

    adjacency, losses, report = run_spd_lasso_symmetry_check_on_rna_ai(
        top_n_genes=None,
        lam=lam,
        gamma=0.01,
        eta=0.5,
        max_iter=100,
    )

    omega, losses_omega = proximal_precision(
        std_data_dict["rna_ai"].T.to_numpy(dtype=float),
        lam=lam,
        gamma=0.01,
        eta=0.5,
        max_iter=1000,
        return_omega=True,
        show_ls_loss=True,
    )

    X = std_data_dict["rna_ai"].T.to_numpy(dtype=float)
    tau = 1 / np.diag(omega)

    f_value = f(omega, X, tau)
    g_value = g(omega, tau, lam)
    total_loss = f_value + g_value

    psd_report_q = test_adjacency_positive_semidefinite(adjacency)
    range_report_q = get_adjacency_element_range(adjacency)

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
