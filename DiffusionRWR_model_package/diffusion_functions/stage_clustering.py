import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from numba import jit

from DiffusionRWR_model_package.graph_generation.generate_multi_graph import (
	create_multigraph,
	create_multigraph_with_layer_transitions,
	create_shadow_network_multigraph,
)
from DiffusionRWR_model_package.graph_generation.preprocess_data import load_data
from DiffusionRWR_model_package.graph_generation.generate_graph_internal import generate_single_layer_graphs
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import (
	cor_gaussian_abs,
	cor_gaussian_abs_inter,
	cor_exponential_abs,
	cor_exponential_abs_inter,
	cor_exponential_shifted,
	intra_layer_corr_exponential_shifted,
	negative_correlation_weight_exponential,
)
from DiffusionRWR_model_package.run_RWR.numba_RWR import preprocess_for_numba_csr


@jit(nopython=True)
def _walk_to_endpoints_core_numba(
	start_idx,
	a_idx,
	b_idx,
	neighbors,
	probs,
	indptr,
	max_steps,
	restart_prob,
):
	path = np.empty(max_steps + 1, dtype=np.int32)
	path[0] = start_idx
	current = start_idx

	for step in range(max_steps):
		if current == a_idx:
			return path[: step + 1], step + 1, 1
		if current == b_idx:
			return path[: step + 1], step + 1, 2

		if restart_prob > 0.0 and np.random.random() < restart_prob:
			return path[: step + 1], step + 1, 3

		start = indptr[current]
		end = indptr[current + 1]
		if start == end:
			return path[: step + 1], step + 1, 0

		node_neighbors = neighbors[start:end]
		node_probs = probs[start:end]

		cumsum = np.empty(len(node_probs), dtype=np.float64)
		cumsum[0] = node_probs[0]
		for i in range(1, len(node_probs)):
			cumsum[i] = cumsum[i - 1] + node_probs[i]

		r = np.random.random()
		idx = np.searchsorted(cumsum, r)
		if idx >= len(node_neighbors):
			idx = len(node_neighbors) - 1

		current = node_neighbors[idx]
		path[step + 1] = current

	return path, max_steps, 0


def build_multigraph_for_endpoints(
	std_data_dict,
	intra_layer_graphs,
	edge_fn_inter,
	start="e5",
	end="e3",
	use_shadow_network=False,
	sign_matrices=None,
	gamma=0.01,
	fix_transition_prob=False,
	alpha=0.1,
):
	if use_shadow_network:
		if sign_matrices is None:
			raise ValueError("sign_matrices must be provided when use_shadow_network=True.")
		return create_shadow_network_multigraph(
			std_data_dict=std_data_dict,
			intra_graphs=intra_layer_graphs,
			sign_matrices=sign_matrices,
			edge_fn_inter=edge_fn_inter,
			gamma=gamma,
			alpha=alpha,
			start=start,
			end=end,
		)

	if fix_transition_prob:
		return create_multigraph_with_layer_transitions(
			std_data_dict=std_data_dict,
			intra_layer_graphs=intra_layer_graphs,
			edge_fn_inter=edge_fn_inter,
			alpha=alpha,
			start=start,
			end=end,
		)

	return create_multigraph(
		std_data_dict=std_data_dict,
		intra_layer_graphs=intra_layer_graphs,
		edge_fn_inter=edge_fn_inter,
		start=start,
		end=end,
	)


def simulate_walks_to_endpoints(
	adjacency_matrix,
	endpoint_a,
	endpoint_b,
	n_walks=10000,
	max_steps=100000,
	min_weight=0.0001,
	restart_prob=0.5,
	seed=None,
	start_nodes=None,
	return_paths=False,
	verbose=True,
):
	"""
	Start each walk from a random node and tally which endpoint is reached.

	For each walk that reaches endpoint A, every node on that path increments
	its A tally by 1 (and likewise for endpoint B).

	If restart_prob > 0, a walk can reset mid-trajectory and be discarded.
	"""
	if restart_prob < 0.0 or restart_prob >= 1.0:
		raise ValueError("restart_prob must be in [0.0, 1.0).")
	if endpoint_a not in adjacency_matrix.index or endpoint_b not in adjacency_matrix.index:
		available = list(adjacency_matrix.index)
		preview = ", ".join(available[:10]) if available else "<none>"
		raise ValueError(
			"Both endpoint_a and endpoint_b must exist in adjacency_matrix index. "
			f"Available nodes (first 10): {preview}"
		)

	nodes = list(adjacency_matrix.index)
	start_pool = nodes if start_nodes is None else list(start_nodes)
	if not start_pool:
		raise ValueError("start_nodes is empty; provide valid node names.")

	if seed is not None:
		np.random.seed(seed)
	rng = np.random.default_rng(seed)

	node_to_idx, idx_to_node, neighbors, probs, indptr = preprocess_for_numba_csr(
		adjacency_matrix, min_weight=min_weight
	)
	a_idx = node_to_idx[endpoint_a]
	b_idx = node_to_idx[endpoint_b]

	tally_a = {node: 0 for node in nodes}
	tally_b = {node: 0 for node in nodes}
	n_a = 0
	n_b = 0
	n_aborted = 0
	n_reset = 0
	n_attempts = 0
	all_paths = [] if return_paths else None

	completed = 0
	while completed < n_walks:
		n_attempts += 1
		start_node = rng.choice(start_pool)
		start_idx = node_to_idx[start_node]
		path_idx, _steps, endpoint_code = _walk_to_endpoints_core_numba(
			start_idx,
			a_idx,
			b_idx,
			neighbors,
			probs,
			indptr,
			max_steps,
			restart_prob,
		)
		if endpoint_code == 3:
			n_reset += 1
			continue

		completed += 1
		path = [idx_to_node[int(idx)] for idx in path_idx]

		if endpoint_code == 1:
			n_a += 1
			for node in path:
				tally_a[node] += 1
		elif endpoint_code == 2:
			n_b += 1
			for node in path:
				tally_b[node] += 1
		else:
			n_aborted += 1

		if return_paths:
			all_paths.append(path)

	tallies_df = pd.DataFrame(
		{
			"tally_A": pd.Series(tally_a),
			"tally_B": pd.Series(tally_b),
		}
	)
	tallies_df["total"] = tallies_df["tally_A"] + tallies_df["tally_B"]
	total_tally = tallies_df["total"].sum()
	if total_tally > 0:
		tallies_df["p_A"] = tallies_df["tally_A"] / total_tally
		tallies_df["p_B"] = tallies_df["tally_B"] / total_tally
	else:
		tallies_df["p_A"] = 0.0
		tallies_df["p_B"] = 0.0

	summary = {
		"n_walks": n_walks,
		"n_attempts": n_attempts,
		"n_reset": n_reset,
		"endpoint_a": endpoint_a,
		"endpoint_b": endpoint_b,
		"n_a": n_a,
		"n_b": n_b,
		"n_aborted": n_aborted,
		"restart_prob": restart_prob,
	}

	if verbose:
		print(
			"Walk summary: "
			f"A={n_a} ({(100 * n_a / n_walks):.1f}%), "
			f"B={n_b} ({(100 * n_b / n_walks):.1f}%), "
			f"aborted={n_aborted} ({(100 * n_aborted / n_walks):.1f}%), "
			f"reset={n_reset} ({(100 * n_reset / max(n_attempts, 1)):.1f}% of attempts)"
		)

	if return_paths:
		return tallies_df, summary, all_paths

	return tallies_df, summary


def run_endpoint_walks_pipeline(
	folder_path=None,
	endpoint_a="e5",
	endpoint_b="e3",
	n_walks=10000,
	max_steps=100000,
	min_weight=0.0001,
	restart_prob=0.05,
	seed=None,
	start_nodes=None,
	return_paths=False,
	verbose=True,
	use_shadow_network=True,
	fix_transition_prob=False,
	alpha=0.1,
	gamma=0.1,
	edge_fn_intra=cor_exponential_abs,
	edge_fn_inter=cor_exponential_abs_inter,
	plot=True,
	return_fig=False,
	marker_size=6,
	plot_trajectories=False,
	n_trajectories=5,
	trajectory_seed=None,
):
	"""
	End-to-end pipeline: load data, build multigraph, and run endpoint walks.
	"""
	if folder_path is None:
		package_root = Path(__file__).resolve().parents[1]
		folder_path = package_root / "data" / "Modelled"
	else:
		folder_path = Path(folder_path)

	data_dict = load_data(str(folder_path))

	cleaned_data_dict = {}
	for key, df in data_dict.items():
		if "k9me2" in key:
			clean_name = "k9me2"
		elif "k20me3" in key:
			clean_name = "k20me3"
		elif "rna" in key:
			clean_name = "rna_ai"
		else:
			clean_name = key
		cleaned_data_dict[clean_name] = df

	if use_shadow_network:
		intra_layer_graphs, sign_matrices = generate_single_layer_graphs(
			cleaned_data_dict,
			edge_fn=edge_fn_intra,
			start=endpoint_a,
			end=endpoint_b,
			return_signs=True,
		)
	else:
		intra_layer_graphs = generate_single_layer_graphs(
			cleaned_data_dict,
			edge_fn=edge_fn_intra,
			start=endpoint_a,
			end=endpoint_b,
		)
		sign_matrices = None

	multi_graph = build_multigraph_for_endpoints(
		std_data_dict=cleaned_data_dict,
		intra_layer_graphs=intra_layer_graphs,
		edge_fn_inter=edge_fn_inter,
		start=endpoint_a,
		end=endpoint_b,
		use_shadow_network=use_shadow_network,
		sign_matrices=sign_matrices,
		gamma=gamma,
		fix_transition_prob=fix_transition_prob,
		alpha=alpha,
	)

	collect_paths = return_paths or (plot and plot_trajectories)
	if collect_paths:
		results = simulate_walks_to_endpoints(
			adjacency_matrix=multi_graph,
			endpoint_a=endpoint_a,
			endpoint_b=endpoint_b,
			n_walks=n_walks,
			max_steps=max_steps,
			min_weight=min_weight,
			restart_prob=restart_prob,
			seed=seed,
			start_nodes=start_nodes,
			return_paths=True,
			verbose=verbose,
		)
		tallies_df, summary, all_paths = results
	else:
		all_paths = None
		results = simulate_walks_to_endpoints(
		adjacency_matrix=multi_graph,
		endpoint_a=endpoint_a,
		endpoint_b=endpoint_b,
		n_walks=n_walks,
		max_steps=max_steps,
		min_weight=min_weight,
		restart_prob=restart_prob,
		seed=seed,
		start_nodes=start_nodes,
		return_paths=False,
		verbose=verbose,
	)
		tallies_df, summary = results

	fig = None
	if plot:
		fig = _plot_endpoint_bias_pca(
			cleaned_data_dict,
			tallies_df,
			marker_size=marker_size,
			use_shadow_network=use_shadow_network,
			endpoint_a=endpoint_a,
			endpoint_b=endpoint_b,
			trajectories=all_paths,
			n_trajectories=n_trajectories,
			trajectory_seed=trajectory_seed,
		)
		fig.write_html("endpoint_bias.html")

	if return_fig:
		return tallies_df, summary, fig

	return tallies_df, summary


def _plot_endpoint_bias_pca(
	data_dict,
	tallies_df,
	marker_size=6,
	use_shadow_network=True,
	endpoint_a=None,
	endpoint_b=None,
	trajectories=None,
	n_trajectories=5,
	trajectory_seed=None,
):
	"""
	Plot PCA points with uniform size. Nodes biased toward A use circles;
	bias toward B use triangles.
	"""
	combined_data = pd.concat(list(data_dict.values()), axis=0)
	pca_model = PCA(n_components=3, random_state=42)
	combined_embedding = pca_model.fit_transform(combined_data)

	embeddings_dict = {}
	start_idx = 0
	for cat_name, cat_data in data_dict.items():
		end_idx = start_idx + len(cat_data)
		embeddings_dict[cat_name] = combined_embedding[start_idx:end_idx]
		start_idx = end_idx

	# Build basis vector coordinates from standardized identity
	standard_basis = np.eye(5)
	standard_basis_std = np.zeros_like(standard_basis)
	for i in range(5):
		row_mean = standard_basis[i].mean()
		row_std = standard_basis[i].std()
		if row_std == 0:
			standard_basis_std[i] = standard_basis[i] - row_mean
		else:
			standard_basis_std[i] = (standard_basis[i] - row_mean) / row_std
	basis_df = pd.DataFrame(standard_basis_std, columns=combined_data.columns)
	basis_coords = pca_model.transform(basis_df)
	basis_names = ["e1", "e2", "e3", "e4", "e5"]

	category_colors = [
		"#1f77b4",
		"#ff7f0e",
		"#2ca02c",
		"#d62728",
		"#9467bd",
		"#8c564b",
		"#e377c2",
		"#7f7f7f",
		"#bcbd22",
		"#17becf",
	]

	fig = go.Figure()

	# Map node names to PCA coordinates (regular, shadow, and basis)
	node_to_coords = {}
	for cat_name, cat_data in data_dict.items():
		cat_embedding = embeddings_dict[cat_name]
		gene_names = list(cat_data.index)
		for i, gene in enumerate(gene_names):
			coords = cat_embedding[i]
			node_to_coords[f"{gene}_{cat_name}"] = coords
			node_to_coords[f"{gene}_{cat_name}_neg"] = coords
	for i, basis_name in enumerate(basis_names):
		node_to_coords[basis_name] = basis_coords[i]

	for idx, (cat_name, cat_data) in enumerate(data_dict.items()):
		cat_embedding = embeddings_dict[cat_name]
		gene_names = list(cat_data.index)
		cat_color = category_colors[idx % len(category_colors)]

		x_vals_a = []
		y_vals_a = []
		z_vals_a = []
		hover_a = []

		x_vals_b = []
		y_vals_b = []
		z_vals_b = []
		hover_b = []

		for i, gene in enumerate(gene_names):
			node_name = f"{gene}_{cat_name}"
			node_name_neg = f"{gene}_{cat_name}_neg"
			tally_a = 0
			tally_b = 0
			if node_name in tallies_df.index:
				row = tallies_df.loc[node_name]
				tally_a += row["tally_A"]
				tally_b += row["tally_B"]
			if use_shadow_network and node_name_neg in tallies_df.index:
				row = tallies_df.loc[node_name_neg]
				tally_a += row["tally_A"]
				tally_b += row["tally_B"]

			is_a = tally_a >= tally_b
			hover_text = (
				f"{gene} ({cat_name})"
				f"<br>Tally A: {tally_a}<br>Tally B: {tally_b}"
			)

			if is_a:
				x_vals_a.append(cat_embedding[i, 0])
				y_vals_a.append(cat_embedding[i, 1])
				z_vals_a.append(cat_embedding[i, 2])
				hover_a.append(hover_text)
			else:
				x_vals_b.append(cat_embedding[i, 0])
				y_vals_b.append(cat_embedding[i, 1])
				z_vals_b.append(cat_embedding[i, 2])
				hover_b.append(hover_text)

		if x_vals_a:
			fig.add_trace(
				go.Scatter3d(
					x=x_vals_a,
					y=y_vals_a,
					z=z_vals_a,
					mode="markers",
					name=f"{cat_name} (A)",
					marker=dict(size=marker_size, color=cat_color, symbol="circle"),
					text=hover_a,
					hovertemplate="%{text}<extra></extra>",
					showlegend=True,
				)
			)

		if x_vals_b:
			fig.add_trace(
				go.Scatter3d(
					x=x_vals_b,
					y=y_vals_b,
					z=z_vals_b,
					mode="markers",
					name=f"{cat_name} (B)",
					marker=dict(size=marker_size, color=cat_color, symbol="diamond"),
					text=hover_b,
					hovertemplate="%{text}<extra></extra>",
					showlegend=True,
				)
			)

	# Basis vectors for reference
	for i, basis_name in enumerate(basis_names):
		fig.add_trace(
			go.Scatter3d(
				x=[basis_coords[i, 0]],
				y=[basis_coords[i, 1]],
				z=[basis_coords[i, 2]],
				mode="markers+text",
				name=basis_name,
				marker=dict(size=marker_size + 4, color="black", symbol="diamond"),
				text=[basis_name],
				textposition="top center",
				hovertemplate=f"{basis_name}<extra></extra>",
				showlegend=True,
			)
		)

	variance_ratios = pca_model.explained_variance_ratio_
	plot_title = "Endpoint Bias PCA Plot"
	if endpoint_a and endpoint_b:
		plot_title = f"Endpoint Bias PCA Plot (A={endpoint_a}, B={endpoint_b})"

	fig.update_layout(
		title=plot_title,
		scene=dict(
			xaxis_title=f"PC1 ({variance_ratios[0]:.2%} variance)",
			yaxis_title=f"PC2 ({variance_ratios[1]:.2%} variance)",
			zaxis_title=f"PC3 ({variance_ratios[2]:.2%} variance)",
			xaxis=dict(showgrid=True, gridcolor="lightgray"),
			yaxis=dict(showgrid=True, gridcolor="lightgray"),
			zaxis=dict(showgrid=True, gridcolor="lightgray"),
		),
		width=1000,
		height=800,
		showlegend=True,
		legend=dict(x=0.02, y=0.98),
	)

	if trajectories and n_trajectories > 0:
		rng = np.random.default_rng(trajectory_seed)
		count = min(n_trajectories, len(trajectories))
		indices = rng.choice(len(trajectories), size=count, replace=False)
		trajectory_colors = [
			"#d62728",
			"#1f77b4",
			"#2ca02c",
			"#ff7f0e",
			"#9467bd",
			"#8c564b",
			"#e377c2",
			"#7f7f7f",
			"#bcbd22",
			"#17becf",
		]
		for i, traj_idx in enumerate(indices):
			path = trajectories[traj_idx]
			x_coords = []
			y_coords = []
			z_coords = []
			for node in path:
				coords = node_to_coords.get(node)
				if coords is None:
					continue
				x_coords.append(coords[0])
				y_coords.append(coords[1])
				z_coords.append(coords[2])

			if len(x_coords) < 2:
				continue

			color = trajectory_colors[i % len(trajectory_colors)]
			fig.add_trace(
				go.Scatter3d(
					x=x_coords,
					y=y_coords,
					z=z_coords,
					mode="lines",
					name=f"Trajectory {i + 1}",
					line=dict(color=color, width=3),
					hoverinfo="skip",
					showlegend=True,
				)
			)

	return fig
