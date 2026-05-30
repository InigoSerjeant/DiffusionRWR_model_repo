from __future__ import annotations

from typing import Callable, Literal

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, normalized_mutual_info_score

from DiffusionRWR_model_package.diffusion_functions.clustering import (
	GraphInput,
	spectral_cluster_graph,
)


ClusteringMethod = Literal["spectral", "custom"]
MutualInfoMethod = Literal["normalized", "adjusted", "raw"]


def cluster_graph(
	graph: GraphInput,
	n_clusters: int,
	method: ClusteringMethod = "spectral",
	custom_clusterer: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None,
	**cluster_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Cluster a single graph using the selected clustering method.

	Parameters
	----------
	graph : GraphInput
		Graph in supported format (adjacency DataFrame, NumPy array, or networkx graph).
	n_clusters : int
		Number of clusters to produce.
	method : {"spectral", "custom"}
		Clustering backend.
	custom_clusterer : callable, optional
		Required when method="custom". Must return (labels_df, affinity_df).
	**cluster_kwargs : dict
		Additional kwargs forwarded to the chosen clustering backend.
	"""
	if method == "spectral":
		return spectral_cluster_graph(graph=graph, n_clusters=n_clusters, **cluster_kwargs)

	if method == "custom":
		if custom_clusterer is None:
			raise ValueError("custom_clusterer must be provided when method='custom'.")
		return custom_clusterer(graph=graph, n_clusters=n_clusters, **cluster_kwargs)

	raise ValueError(f"Unsupported clustering method: {method}")


def cluster_multiple_graphs(
	graphs: dict[str, GraphInput],
	n_clusters: int,
	method: ClusteringMethod = "spectral",
	custom_clusterer: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None,
	**cluster_kwargs,
) -> dict[str, dict[str, pd.DataFrame]]:
	"""
	Cluster multiple graphs with a pluggable clustering backend.

	Parameters
	----------
	graphs : dict[str, GraphInput]
		Mapping from graph name to graph object.
	n_clusters : int
		Number of clusters to produce for each graph.
	method : {"spectral", "custom"}
		Clustering backend to use.
	custom_clusterer : callable, optional
		Required when method="custom". Must return (labels_df, affinity_df).
	**cluster_kwargs : dict
		Additional kwargs passed to clustering function.

	Returns
	-------
	dict[str, dict[str, pd.DataFrame]]
		For each graph key, returns:
		- "labels": DataFrame with node cluster assignments
		- "affinity": DataFrame used for clustering
	"""
	if not isinstance(graphs, dict) or len(graphs) == 0:
		raise ValueError("graphs must be a non-empty dict of {name: graph}.")

	results: dict[str, dict[str, pd.DataFrame]] = {}

	for graph_name, graph in graphs.items():
		labels_df, affinity_df = cluster_graph(
			graph=graph,
			n_clusters=n_clusters,
			method=method,
			custom_clusterer=custom_clusterer,
			**cluster_kwargs,
		)
		results[graph_name] = {
			"labels": labels_df,
			"affinity": affinity_df,
		}

	return results


def _extract_label_series(
	clustering_results: dict[str, dict[str, pd.DataFrame]] | dict[str, pd.DataFrame],
	label_column: str = "cluster",
) -> dict[str, pd.Series]:
	"""
	Extract a label series per graph from either:
	- cluster_multiple_graphs output ({graph: {"labels": df, ...}})
	- direct mapping ({graph: labels_df})
	"""
	label_map: dict[str, pd.Series] = {}

	for graph_name, value in clustering_results.items():
		if isinstance(value, dict):
			if "labels" not in value:
				raise ValueError(f"Graph '{graph_name}' result dict must contain a 'labels' DataFrame.")
			labels_df = value["labels"]
		else:
			labels_df = value

		if not isinstance(labels_df, pd.DataFrame):
			raise TypeError(f"Labels for graph '{graph_name}' must be a pandas DataFrame.")

		if label_column not in labels_df.columns:
			raise ValueError(
				f"Labels DataFrame for graph '{graph_name}' must contain column '{label_column}'."
			)

		label_map[graph_name] = labels_df[label_column].copy()

	return label_map


def calculate_pairwise_mutual_information(
	clustering_results: dict[str, dict[str, pd.DataFrame]] | dict[str, pd.DataFrame],
	method: MutualInfoMethod = "normalized",
	label_column: str = "cluster",
	min_common_nodes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Calculate pairwise mutual information between clusterings from multiple graphs.

	Parameters
	----------
	clustering_results : dict
		Either output of cluster_multiple_graphs(), or mapping {graph_name: labels_df}.
	method : {"normalized", "adjusted", "raw"}
		- normalized: Normalized Mutual Information (NMI, range [0, 1])
		- adjusted: Adjusted Mutual Information (AMI)
		- raw: Mutual Information (MI, unbounded)
	label_column : str
		Column in labels DataFrame containing cluster assignments.
	min_common_nodes : int
		Minimum number of overlapping nodes required to compare two clusterings.

	Returns
	-------
	mi_matrix : pd.DataFrame
		Square matrix of pairwise MI scores with graph names as index/columns.
	mi_table : pd.DataFrame
		Long-form table with columns: graph_a, graph_b, n_common_nodes, mi_score.
	"""
	if min_common_nodes < 1:
		raise ValueError("min_common_nodes must be >= 1.")

	label_map = _extract_label_series(clustering_results, label_column=label_column)
	graph_names = list(label_map.keys())

	if len(graph_names) < 2:
		raise ValueError("At least two graphs are required to compute pairwise mutual information.")

	if method == "normalized":
		score_fn = normalized_mutual_info_score
	elif method == "adjusted":
		score_fn = adjusted_mutual_info_score
	elif method == "raw":
		score_fn = mutual_info_score
	else:
		raise ValueError("method must be one of: 'normalized', 'adjusted', 'raw'.")

	mi_matrix = pd.DataFrame(index=graph_names, columns=graph_names, dtype=float)
	records: list[dict[str, object]] = []

	for i, graph_a in enumerate(graph_names):
		for j, graph_b in enumerate(graph_names):
			if j < i:
				continue

			labels_a = label_map[graph_a]
			labels_b = label_map[graph_b]

			common_nodes = labels_a.index.intersection(labels_b.index)
			n_common = int(len(common_nodes))

			if n_common < min_common_nodes:
				score = float("nan")
			else:
				a = labels_a.loc[common_nodes].astype(int).values
				b = labels_b.loc[common_nodes].astype(int).values
				score = float(score_fn(a, b))

			mi_matrix.loc[graph_a, graph_b] = score
			mi_matrix.loc[graph_b, graph_a] = score

			if i != j:
				records.append(
					{
						"graph_a": graph_a,
						"graph_b": graph_b,
						"n_common_nodes": n_common,
						"mi_score": score,
					}
				)

	mi_table = pd.DataFrame(records).sort_values(["graph_a", "graph_b"]).reset_index(drop=True)
	return mi_matrix, mi_table

