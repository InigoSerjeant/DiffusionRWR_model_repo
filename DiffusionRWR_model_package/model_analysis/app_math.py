from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


try:
    from numba import jit
except Exception:  # pragma: no cover
    jit = None


def is_integer_like(value: str) -> bool:
    try:
        return float(value) % 1 == 0
    except Exception:
        return False


def standardize_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def build_adjacency(df: pd.DataFrame, method: str, n_power: int, sigma: float) -> pd.DataFrame:
    corr = df.T.corr().fillna(0.0)

    if method == "corr_power":
        similarity = 0.5 * (corr + 1.0)
        adjacency = np.abs(similarity.values) ** n_power
    elif method == "cor_gaussian_abs":
        abs_corr = np.abs(corr.values)
        adjacency = np.exp(-0.5 * ((1.0 - abs_corr) / sigma) ** 2)
    else:  # cor_exponential_abs
        abs_corr = np.abs(corr.values)
        adjacency = np.exp(-(1.0 - abs_corr) / sigma)

    np.fill_diagonal(adjacency, 0.0)
    return pd.DataFrame(adjacency, index=df.index, columns=df.index)


def graph_from_adjacency(adjacency: pd.DataFrame, threshold: float, max_nodes: int) -> nx.DiGraph:
    filtered = adjacency.where(adjacency >= threshold, 0.0)

    if max_nodes > 0 and len(filtered) > max_nodes:
        node_strength = filtered.sum(axis=0) + filtered.sum(axis=1)
        selected = node_strength.sort_values(ascending=False).head(max_nodes).index
        filtered = filtered.loc[selected, selected]

    graph = nx.from_pandas_adjacency(filtered, create_using=nx.DiGraph)
    to_remove = [n for n in graph.nodes if graph.in_degree(n) == 0 and graph.out_degree(n) == 0]
    graph.remove_nodes_from(to_remove)
    return graph


def graph_from_adjacency_with_forced_nodes(
    adjacency: pd.DataFrame,
    threshold: float,
    max_nodes: int,
    forced_nodes: list[str],
) -> nx.DiGraph:
    filtered = adjacency.where(adjacency >= threshold, 0.0)

    present_forced = [node for node in forced_nodes if node in filtered.index]

    if max_nodes > 0 and len(filtered) > max_nodes:
        node_strength = filtered.sum(axis=0) + filtered.sum(axis=1)
        top_nodes = node_strength.sort_values(ascending=False).head(max_nodes).index.tolist()
        selected = sorted(set(top_nodes).union(set(present_forced)))
        filtered = filtered.loc[selected, selected]

    return nx.from_pandas_adjacency(filtered, create_using=nx.DiGraph)


def compute_eigenvector_centrality(graph: nx.DiGraph) -> dict[str, float]:
    try:
        return nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
    except Exception:
        return {node: 1.0 for node in graph.nodes}


def compute_pagerank_centrality(graph: nx.DiGraph, alpha: float = 0.85) -> dict[str, float]:
    try:
        return nx.pagerank(graph, alpha=alpha, max_iter=1000, tol=1e-06)
    except Exception:
        return {node: 1.0 for node in graph.nodes}


def compute_degree_centrality(graph: nx.DiGraph) -> dict[str, float]:
    return {node: graph.in_degree(node) + graph.out_degree(node) for node in graph.nodes}


def closest_nodes_to_basis(graph: nx.DiGraph, pos: dict, basis_nodes: list[str]) -> dict[str, str]:
    closest = {}
    non_basis_nodes = [n for n in graph.nodes if n not in basis_nodes]
    if not non_basis_nodes:
        return closest

    for basis in basis_nodes:
        if basis not in pos:
            continue
        basis_xy = np.array(pos[basis])
        best_node = None
        best_dist = float("inf")
        for node in non_basis_nodes:
            node_xy = np.array(pos[node])
            dist = float(np.linalg.norm(node_xy - basis_xy))
            if dist < best_dist:
                best_dist = dist
                best_node = node
        if best_node is not None:
            closest[basis] = best_node
    return closest


def build_multigraph_adjacency_for_clustering(
    std_data_dict: dict[str, pd.DataFrame],
    intra_layer_graphs: dict[str, pd.DataFrame],
    alpha: float,
    inter_method: str,
    inter_n_power: int,
    inter_sigma: float,
    inter_layer_threshold: float,
) -> pd.DataFrame:
    from DiffusionRWR_model_package.graph_generation import edge_weight_functions
    from DiffusionRWR_model_package.graph_generation.generate_multi_graph import create_multigraph_with_layer_transitions

    if not std_data_dict or not intra_layer_graphs:
        return pd.DataFrame()

    edge_fn_base = getattr(edge_weight_functions, inter_method, None)
    if edge_fn_base is None:
        raise ValueError(f"Inter-layer edge function not found: {inter_method}")

    def edge_fn_inter(vec_a, vec_b):
        if inter_method == "inter_layer_corr_power":
            weight = edge_fn_base(vec_a, vec_b, n_power=inter_n_power)
        elif inter_method in {
            "cor_exponential_abs_inter",
            "cor_gaussian_abs_inter",
            "intra_layer_corr_exponential_shifted",
            "intra_layer_corr_gaussian_shifted",
        }:
            weight = edge_fn_base(vec_a, vec_b, sigma=inter_sigma)
        else:
            weight = edge_fn_base(vec_a, vec_b)
        return weight if weight >= inter_layer_threshold else 0.0

    return create_multigraph_with_layer_transitions(
        std_data_dict=std_data_dict,
        intra_layer_graphs=intra_layer_graphs,
        edge_fn_inter=edge_fn_inter,
        alpha=alpha,
        start="e1",
        end="e5",
    )


def build_transition_matrix(adj_matrix: pd.DataFrame, use_unweighted: bool = False) -> pd.DataFrame:
    if use_unweighted:
        transition_matrix = adj_matrix.copy()
        for row_idx in transition_matrix.index:
            out_degree = (transition_matrix.loc[row_idx] > 0).sum()
            if out_degree > 0:
                transition_matrix.loc[row_idx] = (transition_matrix.loc[row_idx] > 0).astype(float) / out_degree
        return transition_matrix

    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return adj_matrix.div(row_sums, axis=0).fillna(0)


if jit is not None:
    @jit(nopython=True)
    def _numba_random_walks(transition_probs, start_idx, n_walks, max_steps, restart_prob):
        n_nodes = transition_probs.shape[0]
        visit_counts = np.zeros(n_nodes)

        for _ in range(n_walks):
            current_idx = start_idx
            for _ in range(max_steps):
                visit_counts[current_idx] += 1

                if np.random.random() < restart_prob:
                    current_idx = start_idx
                else:
                    neighbors = transition_probs[current_idx]
                    has_neighbors = False
                    for i in range(len(neighbors)):
                        if neighbors[i] > 0:
                            has_neighbors = True
                            break

                    if has_neighbors:
                        cumsum = np.cumsum(neighbors)
                        r = np.random.random()
                        for i in range(len(cumsum)):
                            if r < cumsum[i]:
                                current_idx = i
                                break
                    else:
                        current_idx = start_idx

        return visit_counts


def simulate_diffusion_visits(
    graph: nx.DiGraph,
    start_node: str,
    n_walks: int,
    max_steps: int,
    restart_prob: float,
    use_unweighted_transition: bool,
    prefer_numba: bool = True,
) -> tuple[list[str], np.ndarray, str]:
    adj_matrix = nx.to_pandas_adjacency(graph)
    transition_matrix = build_transition_matrix(adj_matrix, use_unweighted=use_unweighted_transition)

    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    start_idx = node_to_idx[start_node]

    message = "Using Python implementation"
    visit_counts = None

    if prefer_numba and jit is not None:
        try:
            trans_matrix_np = transition_matrix.values.astype(np.float64)
            visit_counts = _numba_random_walks(
                trans_matrix_np,
                int(start_idx),
                int(n_walks),
                int(max_steps),
                float(restart_prob),
            )
            message = "Using Numba JIT-compiled random walks"
        except Exception:
            visit_counts = None

    if visit_counts is None:
        visit_counts = np.zeros(len(node_list))
        for _ in range(int(n_walks)):
            current_idx = start_idx
            for _ in range(int(max_steps)):
                visit_counts[current_idx] += 1

                if np.random.random() < restart_prob:
                    current_idx = start_idx
                else:
                    neighbors = transition_matrix.iloc[current_idx]
                    if neighbors.sum() > 0:
                        current_idx = int(np.random.choice(len(node_list), p=neighbors.values))
                    else:
                        current_idx = start_idx

    total = visit_counts.sum()
    if total > 0:
        visit_counts = visit_counts / total

    return node_list, visit_counts, message
