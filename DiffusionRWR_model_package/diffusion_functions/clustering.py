from __future__ import annotations

from typing import Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering


GraphInput = pd.DataFrame | np.ndarray | nx.Graph | nx.DiGraph


def graph_to_adjacency(
	graph: GraphInput,
	node_names: Iterable[str] | None = None,
	weight_attr: str = "weight",
) -> pd.DataFrame:
	"""
	Convert a graph-like object into a square adjacency DataFrame.

	Supported inputs:
	- pd.DataFrame: assumed to be adjacency with matching index/columns
	- np.ndarray: must be square; node_names optional
	- networkx.Graph / networkx.DiGraph
	"""
	if isinstance(graph, pd.DataFrame):
		adjacency = graph.copy()
	elif isinstance(graph, np.ndarray):
		if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
			raise ValueError("NumPy input must be a square 2D array.")
		n_nodes = graph.shape[0]
		if node_names is None:
			index = [f"node_{i}" for i in range(n_nodes)]
		else:
			index = list(node_names)
			if len(index) != n_nodes:
				raise ValueError("Length of node_names must match array dimensions.")
		adjacency = pd.DataFrame(graph, index=index, columns=index)
	elif isinstance(graph, (nx.Graph, nx.DiGraph)):
		adjacency = nx.to_pandas_adjacency(graph, weight=weight_attr)
	else:
		raise TypeError(
			"Unsupported graph type. Expected pd.DataFrame, np.ndarray, nx.Graph, or nx.DiGraph."
		)

	if adjacency.shape[0] != adjacency.shape[1]:
		raise ValueError("Adjacency matrix must be square.")

	if not adjacency.index.equals(adjacency.columns):
		adjacency = adjacency.reindex(index=adjacency.index, columns=adjacency.index, fill_value=0.0)

	return adjacency.astype(float)


def prepare_affinity_matrix(
	adjacency: pd.DataFrame,
	symmetrize: Literal["max", "mean", "min", "none"] = "max",
	clip_negative: bool = True,
	remove_isolated: bool = True,
	add_self_loops: bool = False,
) -> pd.DataFrame:
	"""
	Prepare an adjacency matrix for spectral clustering as a valid affinity matrix.
	"""
	affinity = adjacency.copy().astype(float)

	if clip_negative:
		affinity = affinity.clip(lower=0.0)

	if symmetrize != "none":
		if symmetrize == "max":
			affinity = pd.DataFrame(
				np.maximum(affinity.values, affinity.values.T),
				index=affinity.index,
				columns=affinity.columns,
			)
		elif symmetrize == "mean":
			affinity = (affinity + affinity.T) / 2.0
		elif symmetrize == "min":
			affinity = pd.DataFrame(
				np.minimum(affinity.values, affinity.values.T),
				index=affinity.index,
				columns=affinity.columns,
			)
		else:
			raise ValueError("symmetrize must be one of: 'max', 'mean', 'min', 'none'.")

	np.fill_diagonal(affinity.values, 0.0)

	if remove_isolated:
		strength = affinity.sum(axis=0) + affinity.sum(axis=1)
		keep_nodes = strength[strength > 0].index
		affinity = affinity.loc[keep_nodes, keep_nodes]

	if add_self_loops and not affinity.empty:
		np.fill_diagonal(affinity.values, np.maximum(np.diag(affinity.values), 1e-12))

	if affinity.empty:
		raise ValueError("No nodes remain after preprocessing (all nodes isolated or removed).")

	return affinity


def spectral_cluster_graph(
	graph: GraphInput,
	n_clusters: int,
	node_names: Iterable[str] | None = None,
	weight_attr: str = "weight",
	symmetrize: Literal["max", "mean", "min", "none"] = "max",
	clip_negative: bool = True,
	remove_isolated: bool = True,
	add_self_loops: bool = False,
	assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "kmeans",
	random_state: int = 42,
	n_init: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Cluster nodes of an arbitrary graph using Spectral Clustering.

	Returns
	-------
	labels_df : pd.DataFrame
		Index is node name, columns: cluster, degree_weighted_strength.
	affinity_df : pd.DataFrame
		The affinity matrix actually used by the spectral clustering model.
	"""
	if n_clusters < 2:
		raise ValueError("n_clusters must be >= 2 for spectral clustering.")

	adjacency = graph_to_adjacency(graph, node_names=node_names, weight_attr=weight_attr)
	affinity = prepare_affinity_matrix(
		adjacency,
		symmetrize=symmetrize,
		clip_negative=clip_negative,
		remove_isolated=remove_isolated,
		add_self_loops=add_self_loops,
	)

	if affinity.shape[0] < n_clusters:
		raise ValueError(
			f"n_clusters={n_clusters} is larger than number of available nodes ({affinity.shape[0]})."
		)

	model = SpectralClustering(
		n_clusters=n_clusters,
		affinity="precomputed",
		assign_labels=assign_labels,
		random_state=random_state,
		n_init=n_init,
	)
	labels = model.fit_predict(affinity.values)

	strength = affinity.sum(axis=1)
	labels_df = pd.DataFrame(
		{
			"cluster": labels.astype(int),
			"degree_weighted_strength": strength.values,
		},
		index=affinity.index,
	).sort_values(["cluster", "degree_weighted_strength"], ascending=[True, False])

	return labels_df, affinity


def estimate_clusters_by_eigengap(
	graph: GraphInput,
	k_max: int = 12,
	node_names: Iterable[str] | None = None,
	weight_attr: str = "weight",
	symmetrize: Literal["max", "mean", "min", "none"] = "max",
	clip_negative: bool = True,
	remove_isolated: bool = True,
) -> tuple[int, pd.Series]:
	"""
	Estimate a reasonable cluster count with a normalized-Laplacian eigengap heuristic.

	Returns
	-------
	k_estimate : int
		Estimated number of clusters.
	eigvals : pd.Series
		Sorted eigenvalues used for the estimate.
	"""
	adjacency = graph_to_adjacency(graph, node_names=node_names, weight_attr=weight_attr)
	affinity = prepare_affinity_matrix(
		adjacency,
		symmetrize=symmetrize,
		clip_negative=clip_negative,
		remove_isolated=remove_isolated,
		add_self_loops=False,
	)

	n = affinity.shape[0]
	if n < 3:
		raise ValueError("Need at least 3 nodes to estimate cluster count by eigengap.")

	k_limit = max(2, min(k_max, n - 1))
	degrees = affinity.sum(axis=1).values
	d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
	laplacian = np.eye(n) - d_inv_sqrt @ affinity.values @ d_inv_sqrt

	eigvals = np.linalg.eigvalsh(laplacian)
	eigvals = np.sort(np.real(eigvals))[: k_limit + 1]
	gaps = np.diff(eigvals)
	k_estimate = int(np.argmax(gaps[:k_limit]) + 1)

	return k_estimate, pd.Series(eigvals, name="eigval")
